"""
AppRunner: Orchestrates the complete forecasting pipeline

This class provides a simple interface to run different pipeline configurations
from Databricks notebooks or other execution environments.

Usage:
    from acm_forecast.core import AppRunner
    
    # Simple usage - run all steps
    runner = AppRunner(config_path="path/to/config.yaml")
    runner.run()
    
    # Run specific steps
    runner.run(steps=['load_data', 'prepare_data', 'train_model', 'forecast'])
    
    # From Databricks notebook
    runner = AppRunner(config_path="/dbfs/path/to/config.yaml")
    runner.run(category="Total", output_path="/dbfs/output/forecast.png")
"""
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from pyspark.sql import SparkSession, DataFrame
import pandas as pd

from ..config import AppConfig
from .plugin_registry import PluginFactory
from .interfaces import IModel

logger = logging.getLogger(__name__)


class AppRunner:
    """
    Main application runner for the forecasting pipeline
    
    Orchestrates data loading, preparation, quality checks, training,
    forecasting, evaluation, and visualization based on configuration.
    
    The runner supports flexible step execution, allowing you to run
    only the steps you need for your use case.
    """
    
    def __init__(self, config_path: str, session: Optional[SparkSession] = None):
        """
        Initialize AppRunner with configuration file
        
        Args:
            config_path: Path to YAML configuration file
            session: Optional SparkSession (if None, will be created when needed)
        """
        self.config_path = Path(config_path)
        self.config = AppConfig.from_yaml(str(self.config_path))
        self.spark: Optional[SparkSession] = session
        self.factory: Optional[PluginFactory] = None
        logger.info(f"✅ AppRunner initialized with config: {self.config_path}")
    
    def _get_factory(self) -> PluginFactory:
        """Get or create PluginFactory instance"""
        if self.factory is None:
            self.factory = PluginFactory()
            logger.info("✅ Plugin factory initialized")
        return self.factory
    
    def load_or_generate_data(self) -> DataFrame:
        """
        Step 1: Load data from Delta table or generate sample data
        
        Returns:
            Spark DataFrame with cost data
        """
        logger.info("=" * 70)
        logger.info("STEP 1: Data Loading/Generation")
        logger.info("=" * 70)

        factory = self._get_factory()
        
        # Create data source plugin
        data_source = factory.create_data_source(self.config, spark=self.spark, plugin_name="delta")
        logger.info(f"Loading data from data source: {self.config.data.delta_table_path}")
        df = data_source.load_data()

        if df.isEmpty():
            raise ValueError("Delta table exists but is empty")
        
        logger.info(f"Data columns: {df.columns}")
        logger.info(f"Data schema: {df.printSchema()}")
        
        return df

    def prepare_data(self, raw_df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Step 2: Prepare data (aggregate, clean, transform)
        
        Args:
            raw_df: Raw input DataFrame
            
        Returns:
            Tuple of (agg_df, train_df, test_df) Spark DataFrames
        """
        logger.info("=" * 70)
        logger.info("STEP 2: Data Preparation")
        logger.info("=" * 70)

        factory = self._get_factory()
        
        # Create data preparation plugin
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="default")
        
        # Aggregate daily costs
        logger.info("Aggregating daily costs...")
        agg_df = data_prep.aggregate_data(raw_df)
        logger.info("✅ Aggregated to daily records")
        
        # Split data into train, validation, and test sets
        logger.info("Splitting data...")
        train_df, val_df, test_df = data_prep.split(agg_df)
        logger.info("✅ Split data into training, validation, and test sets")
        
        return train_df, val_df, test_df

    def run_data_quality(self, df: DataFrame) -> dict:
        """
        Step 3: Run data quality validation
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 70)
        logger.info("STEP 3: Data Quality Validation")
        logger.info("=" * 70)

        factory = self._get_factory()
        
        # Create data quality plugin
        data_quality = factory.create_data_quality(self.config, spark=self.spark, plugin_name="default")
        
        # Run comprehensive validation
        logger.info("Running comprehensive data quality checks...")
        validation_results = data_quality.comprehensive_validation(df)
        
        # Log results
        logger.info("Validation Results:")
        for check_name, result in validation_results.items():
            status = "PASS" if result.get("passed", False) else "FAIL"
            logger.info(f"  {check_name}: {status}")
            if "message" in result:
                logger.info(f"    {result['message']}")
        
        return validation_results

    def train_model(self, train_df: DataFrame, category: str = "Total") -> IModel:
        """
        Step 4: Train the forecasting model
        
        Args:
            train_df: Training DataFrame (Spark DataFrame)
            category: Cost category to forecast
            
        Returns:
            Trained model plugin instance (IModel)
        """
        logger.info("=" * 70)
        logger.info("STEP 4: Model Training")
        logger.info("=" * 70)

        factory = self._get_factory()
        
        # Get model name from config
        model_name = self.config.model.selected_model
        logger.info(f"Training {model_name} model for category: {category}")
        
        # Create data preparation plugin for model-specific data prep
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="default")
        
        # Create model plugin
        model = factory.create_model(self.config, category=category, plugin_name=model_name)
        
        # Prepare data for the specific model
        logger.info(f"Preparing data for {model_name} model...")
        if model_name == "xgboost":
            # For XGBoost, use feature engineering
            feature_engineer = factory.create_feature_engineer(self.config, spark=self.spark, plugin_name="default")
            train_data_df = feature_engineer.prepare_xgboost_features(train_df, category=category)
            # Convert to pandas for XGBoost
            train_data_pd = train_data_df.toPandas()
            logger.info(f"Prepared XGBoost data: {len(train_data_pd)} rows")
        else:
            # For Prophet and ARIMA, use data preparation
            train_data_pd = data_prep.prepare_for_training(train_df, model_type=model_name)
            logger.info(f"Prepared {model_name} data: {len(train_data_pd)} rows")
        
        # Train the model
        logger.info("Training model...")
        training_result = model.train(train_data_pd)
        logger.info(f"✅ Model training completed")
        
        return model

    def load_model(self, category: str = "Total") -> IModel:
        """
        Load a trained model from the model registry
        
        Args:
            category: Cost category
            
        Returns:
            Loaded model plugin instance (IModel)
        """
        logger.info("=" * 70)
        logger.info("STEP: Load Model from Registry")
        logger.info("=" * 70)
        
        factory = self._get_factory()
        model_name = self.config.model.selected_model
        
        # Create model registry plugin
        model_registry = factory.create_model_registry(self.config, plugin_name="mlflow")
        
        # Load model from registry
        logger.info(f"Loading {model_name} model for category: {category} from registry...")
        # TODO: Implement model loading from registry
        # model = model_registry.load_model(name=f"{model_name}_{category}")
        
        raise NotImplementedError("Model loading from registry not yet implemented")

    def generate_forecasts(self, model_plugin: IModel) -> pd.DataFrame:
        """
        Step 5: Generate forecasts
        
        Args:
            model_plugin: Trained model plugin instance
            
        Returns:
            DataFrame with forecast results
        """
        logger.info("=" * 70)
        logger.info("STEP 5: Forecast Generation")
        logger.info("=" * 70)
        
        # Get forecast horizon - use forecast_horizons_days[0] if available, else default to 90
        if hasattr(self.config.forecast, 'forecast_horizons_days') and self.config.forecast.forecast_horizons_days:
            horizon_days = self.config.forecast.forecast_horizons_days[0]
        else:
            horizon_days = 90  # Default
        
        model_name = self.config.model.selected_model
        logger.info(f"Generating {horizon_days}-day forecast using {model_name}")
        
        # Use the model plugin's predict method
        logger.info(f"Calling {model_name} model predict method with {horizon_days} periods...")
        forecast_df = model_plugin.predict(periods=horizon_days)
        
        # Convert to pandas if it's a Spark DataFrame
        if hasattr(forecast_df, 'toPandas'):
            forecast_df = forecast_df.toPandas()
        
        logger.info(f"✅ Generated forecast with {len(forecast_df)} predictions")
        
        return forecast_df

    def save_forecasts(self, forecast_df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Save forecast results to file
        
        Args:
            forecast_df: Forecast DataFrame
            output_path: Path to save the forecast (CSV, Parquet, etc.)
        """
        logger.info("=" * 70)
        logger.info("STEP: Save Forecast Results")
        logger.info("=" * 70)
        
        if output_path is None:
            output_path = "forecast_results.csv"
        
        # Save to CSV
        forecast_df.to_csv(output_path, index=False)
        logger.info(f"✅ Forecast results saved to: {output_path}")

    def visualize_forecasts(self, train_df: DataFrame, forecast_df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Step 6: Create forecast visualizations
        
        Args:
            train_df: Training DataFrame
            forecast_df: Forecast DataFrame
            output_path: Optional path to save the plot
        """
        logger.info("=" * 70)
        logger.info("STEP 6: Forecast Visualization")
        logger.info("=" * 70)
        
        import matplotlib.pyplot as plt
        
        # Convert training data to pandas for plotting
        logger.info("Preparing data for visualization...")
        train_pd = train_df.toPandas()
        
        # Ensure date column is datetime
        date_col = self.config.feature.date_column
        target_col = self.config.feature.target_column
        
        train_pd[date_col] = pd.to_datetime(train_pd[date_col])
        
        # Sort by date
        train_pd = train_pd.sort_values(date_col)
        
        # Plot
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(
            train_pd[date_col],
            train_pd[target_col],
            label="Historical Data",
            color="blue",
            linewidth=2
        )
        
        # Plot forecast
        if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
            # Prophet format
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            plt.plot(
                forecast_df['ds'],
                forecast_df['yhat'],
                label="Forecast",
                color="red",
                linewidth=2,
                linestyle="--"
            )
            
            # Confidence intervals
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                plt.fill_between(
                    forecast_df['ds'],
                    forecast_df['yhat_lower'],
                    forecast_df['yhat_upper'],
                    alpha=0.3,
                    color="red",
                    label="Confidence Interval"
                )
        elif 'date' in forecast_df.columns and 'forecast' in forecast_df.columns:
            # Generic format
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            plt.plot(
                forecast_df['date'],
                forecast_df['forecast'],
                label="Forecast",
                color="red",
                linewidth=2,
                linestyle="--"
            )
        
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(f"{target_col} ({self.config.data.cost_categories[0] if self.config.data.cost_categories else 'Cost'})", fontsize=12)
        plt.title(f"Cost Forecast - {self.config.model.selected_model.upper()} Model", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Forecast visualization saved to: {output_path}")
        else:
            output_path = "forecast_visualization.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Forecast visualization saved to: {output_path}")
        
        plt.close()

    def run(self, steps: Optional[List[str]] = None, category: str = "Total", output_path: Optional[str] = None):
        """
        Run the forecasting pipeline with specified steps
        
        Args:
            steps: List of steps to execute. If None, runs all default steps.
                  Valid steps: 
                  - 'load_data': Load or generate data
                  - 'data_quality': Run data quality validation
                  - 'prepare_data': Aggregate and split data
                  - 'train_model': Train the model
                  - 'load_model': Load model from registry (alternative to train_model)
                  - 'forecast': Generate forecasts
                  - 'save_results': Save forecast results
                  - 'visualize': Create visualization
            category: Cost category to forecast (default: "Total")
            output_path: Optional path for output files (visualizations, saved results)
        
        Examples:
            # Example 1: Full training and forecasting pipeline
            runner.run(steps=['load_data', 'data_quality', 'prepare_data', 'train_model', 'forecast'])
            
            # Example 2: Generate sample data, quality check, prepare, train, forecast
            runner.run(steps=['load_data', 'data_quality', 'prepare_data', 'train_model', 'forecast'])
            
            # Example 3: Full pipeline with visualization
            runner.run(steps=['load_data', 'data_quality', 'prepare_data', 'train_model', 
                             'forecast', 'visualize'], output_path="/dbfs/output/forecast.png")
            
            # Example 4: Load existing model and forecast
            runner.run(steps=['load_data', 'load_model', 'forecast', 'save_results', 'visualize'])
            
            # Example 5: Quick forecast with existing model
            runner.run(steps=['load_data', 'load_model', 'forecast', 'visualize'])
        """
        logger.info("=" * 70)
        logger.info("STARTING FORECASTING PIPELINE")
        logger.info("=" * 70)
        
        # Default steps if not specified
        if steps is None:
            steps = ['load_data', 'data_quality', 'prepare_data', 'train_model', 'forecast', 'visualize']
        
        # State variables to pass between steps
        raw_df: Optional[DataFrame] = None
        train_df: Optional[DataFrame] = None
        test_df: Optional[DataFrame] = None
        model_plugin: Optional[IModel] = None
        forecast_df: Optional[pd.DataFrame] = None
        
        try:
            # Step 1: Load or generate data
            if 'load_data' in steps:
                raw_df = self.load_or_generate_data()
            
            # Step 2: Prepare data
            if 'prepare_data' in steps:
                if raw_df is None:
                    raise ValueError("Cannot prepare data: data not loaded. Include 'load_data' step.")
                train_df, val_df, test_df = self.prepare_data(raw_df)
            
            # Step 3: Data quality validation
            if 'data_quality' in steps:
                if train_df is None:
                    if raw_df is None:
                        raw_df = self.load_or_generate_data()
                    # Use raw_df for quality check if daily_df not available
                    self.run_data_quality(raw_df)
                else:
                    self.run_data_quality(train_df)
            
            # Step 4a: Train model
            if 'train_model' in steps:
                if train_df is None or train_df.isEmpty():
                    raise ValueError("Train Dataframe is None or Empty. Include 'load_data' step.")
                model_plugin = self.train_model(train_df, category=category)
            
            # Step 4b: Load model from registry (alternative to training)
            if 'load_model' in steps:
                if model_plugin is None:  # Only load if not already trained
                    model_plugin = self.load_model(category=category)
            
            # Step 5: Generate forecasts
            if 'forecast' in steps:
                if model_plugin is None:
                    raise ValueError("Cannot generate forecast: model not available. Include 'train_model' or 'load_model' step.")
                forecast_df = self.generate_forecasts(model_plugin)
            
            # Step 6: Save forecast results
            if 'save_results' in steps:
                if forecast_df is None:
                    raise ValueError("Cannot save results: forecast not generated. Include 'forecast' step.")
                save_path = output_path if output_path and output_path.endswith('.csv') else "forecast_results.csv"
                self.save_forecasts(forecast_df, output_path=save_path)
            
            # Step 7: Visualize forecasts
            if 'visualize' in steps:
                if forecast_df is None:
                    raise ValueError("Cannot visualize: forecast not generated. Include 'forecast' step.")
                if train_df is None:
                    raise ValueError("Cannot visualize: training data not available. Include 'prepare_data' step.")
                
                viz_path = output_path if output_path and output_path.endswith('.png') else None
                if viz_path is None and output_path:
                    # Use output_path but change extension to .png
                    viz_path = str(Path(output_path).with_suffix('.png'))
                self.visualize_forecasts(train_df, forecast_df, output_path=viz_path)
            
            logger.info("=" * 70)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            # Clean up Spark
            if self.spark is not None:
                self.spark.stop()
                logger.info("Spark session stopped")
