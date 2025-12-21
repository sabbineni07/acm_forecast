"""
Forecast Generation Script
Generates forecasts using trained models from MLflow registry
"""

import logging
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from src.pipeline.forecast_pipeline import ForecastPipeline
from src.config.settings import forecast_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_forecasts(forecasts_dict):
    """Save forecasts to files"""
    import os
    
    output_dir = "forecasts"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for category, forecasts in forecasts_dict.items():
        if 'error' in forecasts:
            continue
        
        for horizon, forecast_df in forecasts.items():
            if isinstance(forecast_df, pd.DataFrame) and len(forecast_df) > 0:
                filename = f"{output_dir}/forecast_{category}_{horizon}_{timestamp}.csv"
                forecast_df.to_csv(filename, index=False)
                logger.info(f"  Saved: {filename}")

def main():
    """Main forecast generation function"""
    
    # Step 1: Initialize Spark Session
    logger.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("ACM_Forecasting_Prediction") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Step 2: Create Forecast Pipeline
        logger.info("Creating forecast pipeline...")
        forecast_pipeline = ForecastPipeline(spark)
        
        # Step 3: Generate forecasts for each category
        categories = ["Total", "Compute", "Storage", "Network", "Database"]
        all_forecasts = {}
        
        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating forecasts for category: {category}")
            logger.info(f"{'='*60}\n")
            
            try:
                # Generate forecasts for multiple horizons
                forecasts = forecast_pipeline.generate_forecasts(
                    category=category,
                    horizons=[30, 90, 180, 365],  # 1, 3, 6, 12 months
                    model_name=None  # Uses best model from registry
                )
                
                all_forecasts[category] = forecasts
                
                logger.info(f"✓ Forecasts generated for {category}")
                
                # Print forecast summary
                for horizon, forecast_df in forecasts.items():
                    if isinstance(forecast_df, pd.DataFrame) and len(forecast_df) > 0:
                        logger.info(
                            f"  {horizon}: {len(forecast_df)} forecast points, "
                            f"Range: ${forecast_df['forecast'].min():.2f} - ${forecast_df['forecast'].max():.2f}"
                        )
                
            except Exception as e:
                logger.error(f"✗ Forecast generation failed for {category}: {e}")
                all_forecasts[category] = {"error": str(e)}
        
        # Step 4: Save forecasts
        logger.info(f"\n{'='*60}")
        logger.info("Saving forecasts...")
        logger.info(f"{'='*60}\n")
        
        save_forecasts(all_forecasts)
        
        # Step 5: Summary
        logger.info(f"\n{'='*60}")
        logger.info("Forecast Generation Summary")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for f in all_forecasts.values() if 'error' not in f)
        logger.info(f"Successfully generated forecasts: {successful}/{len(all_forecasts)} categories")
        
        return all_forecasts
        
    finally:
        # Step 6: Stop Spark
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    forecasts = main()
    print("\n✓ Forecast generation completed!")
    print(f"Forecasts saved to: forecasts/")

