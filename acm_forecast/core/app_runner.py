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
from typing import Optional, List, Tuple, Union, Dict
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
        data_source = factory.create_data_source(self.config, spark=self.spark, plugin_name="acm")
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
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="acm")
        
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
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="acm")
        
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

    def validate_model(self, model_plugin: IModel, val_df: DataFrame) -> Dict[str, float]:
        """
        Validate model performance on validation dataset
        
        Args:
            model_plugin: Trained model plugin instance
            val_df: Validation DataFrame (Spark DataFrame)
            
        Returns:
            Dictionary with validation metrics (MAPE, RMSE, MAE, R²)
        """
        logger.info("=" * 70)
        logger.info("STEP: Model Validation (using val_df)")
        logger.info("=" * 70)
        
        factory = self._get_factory()
        model_name = self.config.model.selected_model
        
        # Prepare validation data for the model
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="acm")
        
        # Convert val_df to model-specific format
        logger.info(f"Preparing validation data for {model_name} model...")
        val_data = data_prep.prepare_for_training(val_df, model_type=model_name)
        
        # Predict on validation period (need to predict len(val_df) periods ahead)
        logger.info(f"Generating predictions for validation period ({len(val_data)} periods)...")
        val_forecast = model_plugin.predict(periods=len(val_data))
        
        # Convert to pandas if needed
        if hasattr(val_forecast, 'toPandas'):
            val_forecast = val_forecast.toPandas()
        
        # Evaluate model performance
        logger.info("Evaluating model performance on validation set...")
        metrics = model_plugin.evaluate(val_forecast, val_data)
        
        logger.info("Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("✅ Model validation completed")
        return metrics
    
    def generate_forecasts(self, model_plugin: IModel) -> Dict[str, pd.DataFrame]:
        """
        Step 5: Generate forecasts for all configured horizons
        
        Args:
            model_plugin: Trained model plugin instance
            
        Returns:
            Dictionary mapping horizon names to forecast DataFrames
            Example: {'30_days': forecast_df_30, '60_days': forecast_df_60, ...}
        """
        logger.info("=" * 70)
        logger.info("STEP 5: Forecast Generation")
        logger.info("=" * 70)
        
        # Get forecast horizons - process ALL values in the array
        if hasattr(self.config.forecast, 'forecast_horizons_days') and self.config.forecast.forecast_horizons_days:
            horizons = self.config.forecast.forecast_horizons_days
        else:
            horizons = [90]  # Default to single 90-day horizon
        
        model_name = self.config.model.selected_model
        logger.info(f"Generating forecasts for horizons: {horizons} days using {model_name}")
        
        forecasts = {}
        
        # Generate forecasts for each horizon
        for horizon_days in horizons:
            logger.info(f"Generating {horizon_days}-day forecast...")
            
            # Use the model plugin's predict method
            forecast_df = model_plugin.predict(periods=horizon_days)
            
            # Convert to pandas if it's a Spark DataFrame
            if hasattr(forecast_df, 'toPandas'):
                forecast_df = forecast_df.toPandas()
            
            # Store with horizon key
            horizon_key = f"{horizon_days}_days"
            forecasts[horizon_key] = forecast_df
            
            logger.info(f"✅ Generated {horizon_days}-day forecast with {len(forecast_df)} predictions")
        
        logger.info(f"✅ Generated forecasts for {len(horizons)} horizon(s)")
        return forecasts

    def save_forecasts(self, forecast_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]], output_path: Optional[str] = None):
        """
        Save forecast results to file
        
        Args:
            forecast_df: Forecast DataFrame or dictionary of forecasts by horizon
            output_path: Path to save the forecast (CSV, Parquet, etc.)
        """
        logger.info("=" * 70)
        logger.info("STEP: Save Forecast Results")
        logger.info("=" * 70)
        
        # Handle dictionary of forecasts (multiple horizons)
        if isinstance(forecast_df, dict):
            for horizon_key, df in forecast_df.items():
                if output_path:
                    # Create horizon-specific filename
                    base_path = Path(output_path).stem
                    save_path = str(Path(output_path).parent / f"{base_path}_{horizon_key}.csv")
                else:
                    save_path = f"forecast_results_{horizon_key}.csv"
                df.to_csv(save_path, index=False)
                logger.info(f"✅ Forecast results saved to: {save_path}")
        else:
            # Single DataFrame
            if output_path is None:
                output_path = "forecast_results.csv"
            forecast_df.to_csv(output_path, index=False)
            logger.info(f"✅ Forecast results saved to: {output_path}")
    
    def evaluate_model(self, model_plugin: IModel, test_df: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test dataset (final evaluation)
        
        Args:
            model_plugin: Trained model plugin instance
            test_df: Test DataFrame (Spark DataFrame) - used for final evaluation
            
        Returns:
            Dictionary with evaluation metrics (MAPE, RMSE, MAE, R²)
        """
        logger.info("=" * 70)
        logger.info("STEP: Model Evaluation (using test_df)")
        logger.info("=" * 70)
        
        factory = self._get_factory()
        model_name = self.config.model.selected_model
        
        # Prepare test data for the model
        data_prep = factory.create_data_preparation(self.config, spark=self.spark, plugin_name="acm")
        
        # Convert test_df to pandas for date extraction
        test_df_pd = test_df.toPandas()
        date_col = self.config.feature.date_column
        test_df_pd[date_col] = pd.to_datetime(test_df_pd[date_col])
        
        # Get test date range
        test_start_date = test_df_pd[date_col].min()
        test_end_date = test_df_pd[date_col].max()
        logger.info(f"Test period: {test_start_date.date()} to {test_end_date.date()}")
        
        # Convert test_df to model-specific format
        logger.info(f"Preparing test data for {model_name} model...")
        test_data = data_prep.prepare_for_training(test_df, model_type=model_name)
        
        # Predict on test period
        # Prophet predict() returns all historical dates + future periods
        logger.info(f"Generating predictions for test period ({len(test_data)} periods)...")
        test_forecast_full = model_plugin.predict(periods=len(test_data))
        
        # Convert to pandas if needed
        if hasattr(test_forecast_full, 'toPandas'):
            test_forecast_full = test_forecast_full.toPandas()
        
        # Extract only test period forecasts (Prophet returns historical + future)
        # For Prophet, 'ds' column contains dates, need to filter to test period
        if 'ds' in test_forecast_full.columns:
            test_forecast_full['ds'] = pd.to_datetime(test_forecast_full['ds'])
            # Filter to test period dates only
            test_forecast = test_forecast_full[
                (test_forecast_full['ds'] >= test_start_date) & 
                (test_forecast_full['ds'] <= test_end_date)
            ].copy()
            logger.info(f"Extracted {len(test_forecast)} forecast points for test period")
        else:
            # If no 'ds' column, assume forecast is already aligned
            test_forecast = test_forecast_full.iloc[-len(test_data):].copy()
            logger.info(f"Using last {len(test_forecast)} forecast points for test")
        
        # Evaluate model performance
        logger.info("Evaluating model performance on test set...")
        metrics = model_plugin.evaluate(test_forecast, test_data)
        
        logger.info("Test Evaluation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("✅ Model evaluation completed")
        return metrics

    def visualize_forecasts(self, train_df: DataFrame, forecasts: Union[pd.DataFrame, Dict[str, pd.DataFrame]], output_path: Optional[str] = None):
        """
        Step 6: Create forecast visualizations
        
        Args:
            train_df: Training DataFrame (Spark DataFrame)
            forecasts: Forecast DataFrame or dictionary of forecasts by horizon
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
        
        # Handle both single DataFrame and dictionary of forecasts
        if isinstance(forecasts, dict):
            # Multiple horizons - plot all
            num_horizons = len(forecasts)
            fig, axes = plt.subplots(num_horizons, 1, figsize=(14, 6 * num_horizons), sharex=True)
            if num_horizons == 1:
                axes = [axes]
            
            for idx, (horizon_key, forecast_df) in enumerate(forecasts.items()):
                ax = axes[idx]
                
                # Plot historical data
                ax.plot(
                    train_pd[date_col],
                    train_pd[target_col],
                    label="Historical Data",
                    color="blue",
                    linewidth=2
                )
                
                # Plot forecast
                if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                    # Prophet format
                    forecast_df = forecast_df.copy()
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                    forecast_df = forecast_df.sort_values('ds')
                    
                    # Extract only future forecasts (not historical fit)
                    last_train_date = train_pd[date_col].max()
                    future_forecast = forecast_df[forecast_df['ds'] > last_train_date]
                    
                    if len(future_forecast) > 0:
                        ax.plot(
                            future_forecast['ds'],
                            future_forecast['yhat'],
                            label=f"Forecast ({horizon_key})",
                            color="red",
                            linewidth=2,
                            linestyle="--"
                        )
                        
                        # Plot confidence intervals if available
                        if 'yhat_lower' in future_forecast.columns and 'yhat_upper' in future_forecast.columns:
                            ax.fill_between(
                                future_forecast['ds'],
                                future_forecast['yhat_lower'],
                                future_forecast['yhat_upper'],
                                alpha=0.2,
                                color="red",
                                label="Confidence Interval"
                            )
                
                ax.set_title(f"Forecast: {horizon_key}")
                ax.set_xlabel("Date")
                ax.set_ylabel(target_col)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            # Single forecast DataFrame - original logic
            forecast_df = forecasts
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
                forecast_df = forecast_df.copy()
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                forecast_df = forecast_df.sort_values('ds')
                
                # Extract only future forecasts (not historical fit)
                last_train_date = train_pd[date_col].max()
                future_forecast = forecast_df[forecast_df['ds'] > last_train_date]
                
                if len(future_forecast) > 0:
                    plt.plot(
                        future_forecast['ds'],
                        future_forecast['yhat'],
                        label="Forecast",
                        color="red",
                        linewidth=2,
                        linestyle="--"
                    )
                    
                    # Plot confidence intervals if available
                    if 'yhat_lower' in future_forecast.columns and 'yhat_upper' in future_forecast.columns:
                        plt.fill_between(
                            future_forecast['ds'],
                            future_forecast['yhat_lower'],
                            future_forecast['yhat_upper'],
                            alpha=0.2,
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
        
        # Save or show plot
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
        val_df: Optional[DataFrame] = None
        test_df: Optional[DataFrame] = None
        model_plugin: Optional[IModel] = None
        forecast_df: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None
        
        try:
            # Step 1: Load or generate data
            if 'load_data' in steps:
                raw_df = self.load_or_generate_data()
            
            # Step 2: Prepare data
            if 'prepare_data' in steps:
                if raw_df is None:
                    raise ValueError("Cannot prepare data: data not loaded. Include 'load_data' step.")
                train_df, val_df, test_df = self.prepare_data(raw_df)
                logger.info(f"Data split - Train: {train_df.count() if train_df else 0} rows, "
                          f"Val: {val_df.count() if val_df else 0} rows, "
                          f"Test: {test_df.count() if test_df else 0} rows")
            
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
            
            # Step 4c: Validate model on validation set
            validation_metrics = None
            if 'validate_model' in steps:
                if model_plugin is None:
                    raise ValueError("Cannot validate: model not available. Include 'train_model' or 'load_model' step.")
                if val_df is None or val_df.isEmpty():
                    logger.warning("Validation DataFrame is None or Empty. Skipping validation step.")
                else:
                    validation_metrics = self.validate_model(model_plugin, val_df)
            
            # Step 5: Generate forecasts
            forecast_df = None
            if 'forecast' in steps:
                if model_plugin is None:
                    raise ValueError("Cannot generate forecast: model not available. Include 'train_model' or 'load_model' step.")
                forecast_df = self.generate_forecasts(model_plugin)
            
            # Step 5b: Evaluate model on test set (final evaluation)
            evaluation_metrics = None
            if 'evaluate_model' in steps:
                if model_plugin is None:
                    raise ValueError("Cannot evaluate: model not available. Include 'train_model' or 'load_model' step.")
                if test_df is None or test_df.isEmpty():
                    logger.warning("Test DataFrame is None or Empty. Skipping evaluation step.")
                else:
                    evaluation_metrics = self.evaluate_model(model_plugin, test_df)
            
            # Step 6: Save forecast results
            if 'save_results' in steps:
                if forecast_df is None:
                    raise ValueError("Cannot save results: forecast not generated. Include 'forecast' step.")
                # Handle both dict and single DataFrame
                if isinstance(forecast_df, dict):
                    # Save each horizon forecast
                    for horizon_key, df in forecast_df.items():
                        save_path = output_path if output_path and output_path.endswith('.csv') else f"forecast_results_{horizon_key}.csv"
                        if output_path:
                            # If output_path provided, create horizon-specific files
                            base_path = Path(output_path).stem
                            save_path = str(Path(output_path).parent / f"{base_path}_{horizon_key}.csv")
                        self.save_forecasts(df, output_path=save_path)
                else:
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
