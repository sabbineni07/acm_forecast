# Step-by-Step Guide: Running Forecast Generation

This guide provides detailed instructions to run the Azure Cost Management Forecasting codebase and generate forecast results.

## Prerequisites

Before running the code, ensure you have:

1. ✅ **Python 3.9+** installed
2. ✅ **Java 8 or 11** installed (for PySpark)
3. ✅ **All dependencies** installed: `pip install -r requirements.txt`
4. ✅ **Access to Databricks Delta table** (or sample data for testing)
5. ✅ **MLflow tracking URI** configured (optional, uses local file system by default)

## Step 1: Environment Setup

### 1.1 Activate Virtual Environment

```bash
# Navigate to project directory
cd /Users/sabbineni/projects/acm

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.2 Set Environment Variables

```bash
# Set Java home (required for PySpark)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Adjust path for your system

# Set MLflow tracking URI (optional - defaults to local file system)
export MLFLOW_TRACKING_URI=file:///tmp/mlruns

# Set Databricks configuration (if using Databricks)
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=your-token
```

### 1.3 Verify Installation

```bash
python -c "from acm_forecast.pipeline.training_pipeline import TrainingPipeline; print('✓ Imports successful')"
```

## Step 2: Prepare Data

### 2.1 Option A: Using Databricks Delta Table (Production)

If you have a Databricks job that extracts data to a Delta table:

```python
# The code will automatically connect to:
# - Database: cost_management
# - Table: amortized_costs
# - Path: azure_cost_management.amortized_costs
#
# This uses the data_source module:
# from acm_forecast.data.data_source import DataSource
# data_source = DataSource(spark)
# df = data_source.load_from_delta(start_date="2023-01-01", end_date="2024-01-01")
```

### 2.2 Option B: Using Sample Data (Testing)

If you want to test with sample data first:

```python
# Create a test script: test_forecast.py
from pyspark.sql import SparkSession
from acm_forecast.data.data_source import DataSource
from acm_forecast.data.data_preparation import DataPreparation
from acm_forecast.data.data_quality import DataQualityValidator
import pandas as pd

# Initialize Spark
spark = SparkSession.builder \
    .appName("ACM_Forecast_Test") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Load sample data (if you have CSV files)
sample_data = spark.read.csv("data/sample_azure_costs.csv", header=True, inferSchema=True)

# Save to Delta format for testing (optional)
sample_data.write.format("delta").mode("overwrite").saveAsTable("cost_management.test_amortized_costs")

# Test data quality validation
quality_validator = DataQualityValidator(spark)
quality_results = quality_validator.comprehensive_validation(sample_data)
print(f"Data quality score: {quality_results['quality_score']:.2f}%")
```

## Step 3: Run Training Pipeline

### 3.1 Understanding the Pipeline Architecture

The training pipeline uses the following modular components:

1. **`data_source`** (`acm_forecast/data/data_source.py`): 
   - Loads data from Databricks Delta tables
   - Handles data mapping and validation
   - Used in: `TrainingPipeline.data_source.load_from_delta()`

2. **`data_quality`** (`acm_forecast/data/data_quality.py`):
   - Validates data completeness, accuracy, consistency, and timeliness
   - Calculates data quality scores
   - Used in: `TrainingPipeline.data_quality.comprehensive_validation()`

3. **`data_preparation`** (`acm_forecast/data/data_preparation.py`):
   - Aggregates daily costs
   - Handles missing values and outliers
   - Splits data into train/validation/test sets
   - Prepares data for Prophet and ARIMA models
   - Used in: `TrainingPipeline.data_prep.*`

4. **`feature_engineering`** (`acm_forecast/data/feature_engineering.py`):
   - Creates temporal features (year, month, day, cyclical encoding)
   - Creates lag features (1, 2, 3, 7, 14, 30 days)
   - Creates rolling window features (mean, std, min, max)
   - Prepares features for XGBoost model
   - Used in: `TrainingPipeline.feature_engineer.prepare_xgboost_features()`

5. **`models`** (`acm_forecast/models/`):
   - **ProphetForecaster**: Facebook Prophet model
   - **ARIMAForecaster**: ARIMA/SARIMA model
   - **XGBoostForecaster**: XGBoost gradient boosting model
   - Used in: `TrainingPipeline.prophet_forecaster`, `arima_forecaster`, `xgboost_forecaster`

6. **`evaluation`** (`acm_forecast/evaluation/`):
   - **ModelEvaluator**: Calculates performance metrics (RMSE, MAE, MAPE, R²)
   - **ModelComparator**: Compares multiple models and selects best
   - Used in: `TrainingPipeline.evaluator`, `TrainingPipeline.comparator`

### 3.2 Create Training Script

Create a file `acm_forecast/examples/run_training.py`:

```python
"""
Training Pipeline Script
Generates forecasts by training models on historical data
"""

import logging
from pyspark.sql import SparkSession
from acm_forecast.pipeline.training_pipeline import TrainingPipeline
from acm_forecast.config.settings import data_config

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
```

### 3.2 Run Training

```bash
python acm_forecast/examples/run_training.py
```

**Expected Output:**
```
2024-01-15 10:00:00 - __main__ - INFO - Initializing Spark session...
2024-01-15 10:00:05 - __main__ - INFO - Creating training pipeline...
2024-01-15 10:00:05 - __main__ - INFO - ============================================================
2024-01-15 10:00:05 - __main__ - INFO - Training models for category: Total
2024-01-15 10:00:05 - __main__ - INFO - ============================================================
2024-01-15 10:00:05 - src.pipeline.training_pipeline - INFO - Starting training pipeline for Total
2024-01-15 10:00:05 - src.pipeline.training_pipeline - INFO - Step 1: Loading data from Delta table
2024-01-15 10:00:10 - src.data.data_source - INFO - Loading data from azure_cost_management.amortized_costs
2024-01-15 10:00:15 - src.pipeline.training_pipeline - INFO - Step 2: Validating data quality
2024-01-15 10:00:16 - src.data.data_quality - INFO - Performing comprehensive data quality validation
2024-01-15 10:00:20 - src.pipeline.training_pipeline - INFO - Step 3: Aggregating daily costs
2024-01-15 10:00:25 - src.pipeline.training_pipeline - INFO - Step 4: Preparing data
2024-01-15 10:00:26 - src.pipeline.training_pipeline - INFO - Step 5: Splitting data
2024-01-15 10:00:27 - src.pipeline.training_pipeline - INFO - Step 6: Training models
2024-01-15 10:00:27 - src.pipeline.training_pipeline - INFO - Training Prophet model
2024-01-15 10:00:30 - src.models.prophet_model - INFO - Training Prophet model for Total with 365 records
2024-01-15 10:01:00 - src.pipeline.training_pipeline - INFO - Training ARIMA model
2024-01-15 10:01:05 - src.models.arima_model - INFO - Training ARIMA model for Total with 365 records
2024-01-15 10:02:00 - src.pipeline.training_pipeline - INFO - Training XGBoost model
2024-01-15 10:02:05 - src.data.feature_engineering - INFO - Prepared 25 features for XGBoost
2024-01-15 10:02:10 - src.models.xgboost_model - INFO - Training XGBoost model for Total
2024-01-15 10:03:00 - src.evaluation.model_evaluator - INFO - Evaluated Prophet: MAPE=8.50%, R²=0.8500
...
```

## Step 4: Generate Forecasts

### 4.1 Create Forecast Script

Create a file `acm_forecast/examples/run_forecast.py`:

```python
"""
Forecast Generation Script
Generates forecasts using trained models from MLflow registry
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from acm_forecast.pipeline.forecast_pipeline import ForecastPipeline
from acm_forecast.registry.model_registry import ModelRegistry
from acm_forecast.config.settings import forecast_config, registry_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    forecasts = main()
    print("\n✓ Forecast generation completed!")
    print(f"Forecasts saved to: forecasts/")
```

### 4.2 Run Forecast Generation

```bash
python acm_forecast/examples/run_forecast.py
```

## Step 5: Complete End-to-End Script

For a complete workflow, create `acm_forecast/examples/run_complete_pipeline.py`:

```python
"""
Complete End-to-End Pipeline
Trains models and generates forecasts in one run
"""

import logging
import sys
from pyspark.sql import SparkSession
from acm_forecast.pipeline.training_pipeline import TrainingPipeline
from acm_forecast.pipeline.forecast_pipeline import ForecastPipeline
from acm_forecast.config.settings import data_config

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
            horizons=[30, 90, 180, 365]
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
```

## Step 6: Running the Complete Pipeline

```bash
# Run complete pipeline
python acm_forecast/examples/run_complete_pipeline.py

# Check logs
tail -f forecast_pipeline.log
```

## Step 7: View Results

### 7.1 Check MLflow UI

```bash
# Start MLflow UI to view registered models
mlflow ui --port 5000

# Open browser to: http://localhost:5000
```

### 7.2 View Forecast Files

```bash
# List generated forecasts
ls -lh forecasts/

# View a forecast file
head -20 forecasts/forecast_Total_30_days_*.csv
```

### 7.3 Create Visualization Script

Create `visualize_forecasts.py`:

```python
"""
Visualize Forecast Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import os

def load_forecasts(category="Total", horizon="30_days"):
    """Load forecast files"""
    pattern = f"forecasts/forecast_{category}_{horizon}_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No forecast files found for {category} - {horizon}")
        return None
    
    # Load most recent
    latest_file = max(files, key=os.path.getctime)
    return pd.read_csv(latest_file)

def visualize_forecast(category="Total", horizon="30_days"):
    """Create interactive forecast visualization"""
    df = load_forecasts(category, horizon)
    
    if df is None:
        return
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='blue', width=2)
    ))
    
    # Add confidence intervals if available
    if 'lower' in df.columns and 'upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['lower'],
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(width=0)
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Forecast for {category} - {horizon}',
        xaxis_title='Date',
        yaxis_title='Cost ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Save and show
    output_file = f"forecasts/visualization_{category}_{horizon}.html"
    fig.write_html(output_file)
    print(f"Visualization saved to: {output_file}")
    
    return fig

if __name__ == "__main__":
    # Visualize forecasts
    for category in ["Total", "Compute", "Storage"]:
        for horizon in ["30_days", "90_days", "180_days"]:
            visualize_forecast(category, horizon)
    
    print("\n✓ Visualizations created!")
```

Run visualization:

```bash
python visualize_forecasts.py
```

## Quick Reference: Command Summary

```bash
# 1. Setup
source venv/bin/activate
export JAVA_HOME=/path/to/java

# 2. Training only
python acm_forecast/examples/run_training.py

# 3. Forecast generation only (requires trained models)
python acm_forecast/examples/run_forecast.py

# 4. Complete pipeline (training + forecasting)
python acm_forecast/examples/run_complete_pipeline.py

# 5. View MLflow models
mlflow ui --port 5000

# 6. Visualize results
python visualize_forecasts.py
```

## Troubleshooting

### Issue: "No module named 'src'"

**Solution:**
```bash
# Ensure you're in the project root directory
cd /Users/sabbineni/projects/acm

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "Java gateway process exited"

**Solution:**
```bash
# Verify Java is installed
java -version

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Issue: "Delta table not found"

**Solution:**
- Verify Databricks connection
- Check table path in `acm_forecast/config/settings.py`
- Use sample data for testing (see Step 2.2)

### Issue: "MLflow model not found"

**Solution:**
- Ensure models are registered (run training first)
- Check MLflow tracking URI
- Verify model name in registry

## Next Steps

1. **Customize Configuration**: Edit `acm_forecast/config/settings.py` for your needs
2. **Schedule Jobs**: Set up cron jobs or Databricks jobs for automated runs
3. **Monitor Performance**: Use monitoring modules in `acm_forecast/monitoring/`
4. **Retrain Models**: Schedule retraining using `acm_forecast/monitoring/retraining_scheduler.py`

## Additional Resources

- See `MODEL_DOCUMENTATION.md` for model details
- See `acm_forecast/README.md` for code structure
- See `INSTALLATION.md` for setup instructions

