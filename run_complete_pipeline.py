"""
Complete End-to-End Pipeline
Trains models and generates forecasts in one run
"""

import logging
import sys
import pandas as pd
from pyspark.sql import SparkSession
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.forecast_pipeline import ForecastPipeline
from src.config.settings import data_config, forecast_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecast_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Complete pipeline execution"""
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ACM_Complete_Pipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # ============================================================
        # PHASE 1: TRAINING
        # ============================================================
        logger.info("="*60)
        logger.info("PHASE 1: MODEL TRAINING")
        logger.info("="*60)
        
        training_pipeline = TrainingPipeline(spark)
        
        # Train for Total category (you can add more categories)
        training_results = training_pipeline.run(
            category="Total",
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        
        logger.info("✓ Training phase completed")
        
        # ============================================================
        # PHASE 2: FORECAST GENERATION
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: FORECAST GENERATION")
        logger.info("="*60)
        
        forecast_pipeline = ForecastPipeline(spark)
        
        # Generate forecasts
        forecasts = forecast_pipeline.generate_forecasts(
            category="Total",
            horizons=forecast_config.forecast_horizons_days
        )
        
        logger.info("✓ Forecast generation completed")
        
        # ============================================================
        # PHASE 3: RESULTS SUMMARY
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: RESULTS SUMMARY")
        logger.info("="*60)
        
        # Print training results
        logger.info("\nTraining Results:")
        for model_name, model_results in training_results.items():
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                logger.info(
                    f"  {model_name}: "
                    f"MAPE={metrics.get('mape', 'N/A'):.2f}%, "
                    f"R²={metrics.get('r2', 'N/A'):.4f}"
                )
        
        # Print forecast summary
        logger.info("\nForecast Summary:")
        for horizon, forecast_df in forecasts.items():
            if isinstance(forecast_df, pd.DataFrame) and len(forecast_df) > 0:
                logger.info(
                    f"  {horizon}: "
                    f"{len(forecast_df)} points, "
                    f"Total Forecast: ${forecast_df['forecast'].sum():,.2f}"
                )
        
        return {
            "training": training_results,
            "forecasts": forecasts
        }
        
    finally:
        spark.stop()

if __name__ == "__main__":
    results = main()
    print("\n" + "="*60)
    print("✓ Complete pipeline executed successfully!")
    print("="*60)

