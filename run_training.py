"""
Training Pipeline Script
Generates forecasts by training models on historical data
"""

import logging
from pyspark.sql import SparkSession
from src.pipeline.training_pipeline import TrainingPipeline
from src.config.settings import data_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    
    # Step 1: Initialize Spark Session
    logger.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("ACM_Forecasting_Training") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")  # Reduce Spark logging
    
    try:
        # Step 2: Create Training Pipeline
        logger.info("Creating training pipeline...")
        pipeline = TrainingPipeline(spark)
        
        # Step 3: Train models for each category
        categories = ["Total"] + data_config.cost_categories
        
        all_results = {}
        
        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for category: {category}")
            logger.info(f"{'='*60}\n")
            
            try:
                # Run training pipeline
                results = pipeline.run(
                    category=category,
                    start_date="2023-01-01",  # Adjust based on your data
                    end_date="2024-01-01"     # Adjust based on your data
                )
                
                all_results[category] = results
                
                logger.info(f"✓ Training completed for {category}")
                
                # Print performance metrics
                for model_name, model_results in results.items():
                    if 'metrics' in model_results:
                        metrics = model_results['metrics']
                        logger.info(
                            f"  {model_name}: "
                            f"MAPE={metrics.get('mape', 'N/A'):.2f}%, "
                            f"R²={metrics.get('r2', 'N/A'):.4f}"
                        )
                
            except Exception as e:
                logger.error(f"✗ Training failed for {category}: {e}")
                all_results[category] = {"error": str(e)}
        
        # Step 4: Summary
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for r in all_results.values() if 'error' not in r)
        logger.info(f"Successfully trained: {successful}/{len(all_results)} categories")
        
        return all_results
        
    finally:
        # Step 5: Stop Spark
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    results = main()
    print("\n✓ Training pipeline completed!")

