"""
End-to-End Example: Complete Forecasting Pipeline

This example demonstrates the complete workflow using the plugin architecture:
1. Read data from Delta data source (or generate sample data)
2. Prepare the data
3. Run data quality checks
4. Train the model
5. Generate forecasts
6. Visualize results

Usage:
    python -m acm_forecast.examples.run_end_to_end
    # Or with custom config:
    python -m acm_forecast.examples.run_end_to_end --config config_end_to_end.yaml
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from acm_forecast.config import AppConfig
from acm_forecast.core import PluginFactory
from acm_forecast.core.interfaces import IModel
from acm_forecast.evaluation.performance_metrics import PerformanceMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_spark() -> SparkSession:
    """Set up Spark session with Delta Lake support"""
    from delta import configure_spark_with_delta_pip
    
    builder = SparkSession.builder \
        .appName("ACM_Forecast_E2E") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g")
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_or_generate_data(
    factory: PluginFactory,
    config: AppConfig,
    spark: SparkSession
) -> DataFrame:
    """
    Step 1: Load data from Delta table or generate sample data
    
    Args:
        factory: PluginFactory instance
        config: AppConfig instance
        spark: SparkSession
        
    Returns:
        Spark DataFrame with cost data
    """
    logger.info("=" * 70)
    logger.info("STEP 1: Data Loading/Generation")
    logger.info("=" * 70)
    
    # Create data source plugin
    data_source = factory.create_data_source(config, spark=spark, plugin_name="delta")
    logger.info(f"Loading data from data source: {config.data.delta_table_path}")
    df = data_source.load_data()

    if df.isEmpty():
        raise ValueError("Delta table exists but is empty")
    
    logger.info(f"Data columns: {df.columns}")
    logger.info(f"Data schema: {df.printSchema()}")
    
    return df


def prepare_data(
    factory: PluginFactory,
    config: AppConfig,
    spark: SparkSession,
    raw_df: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Step 2: Prepare data (aggregate, clean, transform)
    
    Args:
        factory: PluginFactory instance
        config: AppConfig instance
        spark: SparkSession
        raw_df: Raw input DataFrame
        
    Returns:
        Prepared DataFrame
    """
    logger.info("=" * 70)
    logger.info("STEP 2: Data Preparation")
    logger.info("=" * 70)
    
    # Create data preparation plugin
    data_prep = factory.create_data_preparation(config, spark=spark, plugin_name="default")
    
    # Aggregate daily costs
    logger.info("Aggregating daily costs...")
    daily_df = data_prep.aggregate_data(raw_df)
    logger.info("✅ Aggregated to daily records")
    
    # Split data into train, validation, and test sets
    logger.info("Splitting data...")
    train_df, val_df, test_df = data_prep.split(daily_df)
    logger.info("✅ Split data into training, validation, and test sets")
    
    return daily_df, train_df, test_df


def run_data_quality(
    factory: PluginFactory,
    config: AppConfig,
    spark: SparkSession,
    df: DataFrame
) -> dict:
    """
    Step 3: Run data quality validation
    
    Args:
        factory: PluginFactory instance
        config: AppConfig instance
        spark: SparkSession
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    logger.info("=" * 70)
    logger.info("STEP 3: Data Quality Validation")
    logger.info("=" * 70)
    
    # Create data quality plugin
    data_quality = factory.create_data_quality(config, spark=spark, plugin_name="default")
    
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


def train_model(
    factory: PluginFactory,
    config: AppConfig,
    spark: SparkSession,
    train_df: DataFrame,  # Spark DataFrame
    category: str = "Total"
) -> IModel:
    """
    Step 4: Train the forecasting model
    
    Args:
        factory: PluginFactory instance
        config: AppConfig instance
        spark: SparkSession
        train_df: Training DataFrame
        category: Cost category to forecast
        
    Returns:
        Trained model plugin instance (IModel) containing the trained model.
        The model can be used to generate forecasts via the predict() method.
    """
    logger.info("=" * 70)
    logger.info("STEP 4: Model Training")
    logger.info("=" * 70)
    
    # Get model name from config
    model_name = config.model.selected_model
    logger.info(f"Training {model_name} model for category: {category}")
    
    # Create data preparation plugin for model-specific data prep
    data_prep = factory.create_data_preparation(config, spark=spark, plugin_name="default")
    
    # Create model plugin
    model = factory.create_model(config, category=category, plugin_name=model_name)
    
    # Prepare data for the specific model
    logger.info(f"Preparing data for {model_name} model...")
    if model_name == "xgboost":
        # For XGBoost, use feature engineering
        feature_engineer = factory.create_feature_engineer(config, spark=spark, plugin_name="default")
        train_data_df = feature_engineer.prepare_xgboost_features(train_df, category=category)
        # Convert to pandas for XGBoost
        train_data_pd = train_data_df.toPandas()
        logger.info(f"Prepared XGBoost data: {len(train_data_pd)} rows")
    else:
        # For Prophet and ARIMA, use data preparation
        train_data_pd = data_prep.prepare_for_training(train_df, model_type=model_name)
        logger.info(f"Prepared {model_name} data: {len(train_data_pd)} rows")
    
    # Train the model - model.train() trains the model and stores it in model.model
    logger.info("Training model...")
    training_result = model.train(train_data_pd)
    logger.info(f"model Training completed: {training_result}")
    
    # Model plugin now contains the trained model (accessible via model.model)
    return model


def generate_forecasts(
    config: AppConfig,
    model_plugin: IModel
) -> pd.DataFrame:
    """
    Step 5: Generate forecasts
    
    Args:
        config: AppConfig instance
        model_plugin: Trained model plugin instance
        
    Returns:
        DataFrame with forecast results
    """
    logger.info("=" * 70)
    logger.info("STEP 5: Forecast Generation")
    logger.info("=" * 70)
    
    # Get forecast horizon - use forecast_horizons_days[0] if available, else default to 90
    if hasattr(config.forecast, 'forecast_horizons_days') and config.forecast.forecast_horizons_days:
        horizon_days = config.forecast.forecast_horizons_days[0]
    else:
        horizon_days = 90  # Default
    
    model_name = config.model.selected_model
    logger.info(f"Generating {horizon_days}-day forecast using {model_name}")
    
    # Use the model plugin's predict method
    # The predict method signature: predict(periods)
    # Model plugin is already trained (model.model contains the trained model)
    logger.info(f"Calling {model_name} model predict method with {horizon_days} periods...")
    forecast_df = model_plugin.predict(periods=horizon_days)
    
    # Convert to pandas if it's a Spark DataFrame
    if hasattr(forecast_df, 'toPandas'):
        forecast_df = forecast_df.toPandas()
    
    logger.info(f"✅ Generated forecast with {len(forecast_df)} predictions")
    
    return forecast_df


def visualize_forecasts(train_df: DataFrame, forecast_df: pd.DataFrame, config: AppConfig,
                        output_path: Optional[str] = None):
    """
    Step 6: Create forecast visualizations
    
    Args:
        train_df: Training DataFrame
        forecast_df: Forecast DataFrame
        config: AppConfig instance
        output_path: Optional path to save the plot
    """
    logger.info("=" * 70)
    logger.info("STEP 6: Forecast Visualization")
    logger.info("=" * 70)
    
    # Convert training data to pandas for plotting
    logger.info("Preparing data for visualization...")
    train_pd = train_df.toPandas()
    
    # Ensure date column is datetime
    date_col = config.feature.date_column
    target_col = config.feature.target_column
    
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
    plt.ylabel(f"{target_col} ({config.data.cost_categories[0] if config.data.cost_categories else 'Cost'})", fontsize=12)
    plt.title(f"Cost Forecast - {config.model.selected_model.upper()} Model", fontsize=14, fontweight="bold")
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


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run end-to-end forecasting pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="acm_forecast/examples/config_end_to_end.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Total",
        help="Cost category to forecast"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forecast_visualization.png",
        help="Output path for forecast visualization"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Try relative to examples directory
            examples_dir = Path(__file__).parent
            config_path = examples_dir / args.config
        
        config = AppConfig.from_yaml(str(config_path))
        logger.info(f"✅ Configuration loaded from: {config_path}")
        
        # Initialize Spark
        spark = setup_spark()
        logger.info("✅ Spark session initialized")
        
        # Initialize plugin factory
        factory = PluginFactory()
        logger.info("✅ Plugin factory initialized")
        
        # Step 1: Load or generate data
        raw_df = load_or_generate_data(factory, config, spark)
        
        # Step 2: Prepare data
        daily_df, train_df, test_df = prepare_data(factory, config, spark, raw_df)
        
        # Step 3: Run data quality
        validation_results = run_data_quality(factory, config, spark, daily_df)
        
        # Step 4: Train model
        model_plugin = train_model(factory, config, spark, train_df, category=args.category)
        
        # Step 5: Generate forecasts
        forecast_df = generate_forecasts(config, model_plugin)
        
        # Step 6: Visualize forecasts
        visualize_forecasts(train_df, forecast_df, config, output_path=args.output)
        
        logger.info("=" * 70)
        logger.info("✅ END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up Spark
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()

