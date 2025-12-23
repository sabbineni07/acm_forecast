"""
Example: Using YAML-based Configuration
Demonstrates how to use the YAML configuration system
"""

from acm_forecast.config import AppConfig
from pyspark.sql import SparkSession
from acm_forecast.pipeline.training_pipeline import TrainingPipeline

def main():
    """Example of using YAML configuration"""
    
    # Option 1: Load default config (from acm_forecast/config/config.yaml)
    print("="*60)
    print("Loading Configuration from YAML")
    print("="*60)
    
    config = AppConfig.from_yaml()  # Loads from acm_forecast/config/config.yaml
    
    # Access different config sections
    print(f"\nData Configuration:")
    print(f"  Delta Table: {config.data.delta_table_path}")
    print(f"  Database: {config.data.database_name}")
    print(f"  Primary Region: {config.data.primary_region} ({config.data.primary_region_weight*100}%)")
    
    print(f"\nModel Configuration:")
    print(f"  Prophet - Yearly Seasonality: {config.model.prophet.yearly_seasonality}")
    print(f"  ARIMA - Seasonal Period: {config.model.arima.seasonal_period}")
    print(f"  XGBoost - N Estimators: {config.model.xgboost.n_estimators}")
    
    print(f"\nTraining Configuration:")
    print(f"  Train Split: {config.training.train_split*100}%")
    print(f"  Validation Split: {config.training.validation_split*100}%")
    print(f"  Test Split: {config.training.test_split*100}%")
    
    print(f"\nFeature Configuration:")
    print(f"  Lag Periods: {config.feature.lag_periods}")
    print(f"  Rolling Windows: {config.feature.rolling_windows}")
    
    print(f"\nPerformance Targets:")
    print(f"  Target MAPE: {config.performance.target_mape}%")
    print(f"  Target R²: {config.performance.target_r2}")
    
    print(f"\nForecast Configuration:")
    print(f"  Forecast Horizons: {config.forecast.forecast_horizons_days} days")
    
    # Option 2: Load custom config file
    print("\n" + "="*60)
    print("Loading Custom Configuration")
    print("="*60)
    
    # custom_config = AppConfig("path/to/custom_config.yaml")
    
    # Option 3: Use in pipeline
    print("\n" + "="*60)
    print("Using Config in Training Pipeline")
    print("="*60)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("YAML_Config_Example") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    try:
        # Create pipeline (will use config from AppConfig)
        pipeline = TrainingPipeline(spark)
        
        # Run training with config-driven settings
        print(f"\nRunning training for category: Total")
        print(f"Using data from: {config.data.delta_table_path}")
        print(f"Training split: {config.training.train_split*100}%")
        
        # Uncomment to actually run training:
        # results = pipeline.run(
        #     category="Total",
        #     start_date="2023-01-01",
        #     end_date="2024-01-01"
        # )
        
        print("\n✓ Configuration loaded and ready to use!")
        
    finally:
        spark.stop()
    
    # Option 4: Reload config if file changes
    print("\n" + "="*60)
    print("Reloading Configuration")
    print("="*60)
    print("(Make changes to config.yaml, then call AppConfig.from_yaml() again)")
    # config = AppConfig.from_yaml()  # Reload from file

if __name__ == "__main__":
    main()

