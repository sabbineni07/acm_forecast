# Module Usage Guide

This document explains how each module in the codebase is used in the forecast generation pipeline.

## Module Overview

The codebase is organized into modular components that work together:

```
acm_forecast/
├── data/              # Data processing modules
│   ├── data_source.py          # Loads data from Delta tables
│   ├── data_quality.py         # Validates data quality
│   ├── data_preparation.py     # Prepares and cleans data
│   └── feature_engineering.py  # Creates features for ML models
├── models/            # Forecasting models
│   ├── prophet_model.py        # Prophet time series model
│   ├── arima_model.py          # ARIMA time series model
│   └── xgboost_model.py        # XGBoost ML model
├── evaluation/        # Model evaluation
│   ├── performance_metrics.py  # Calculates metrics
│   ├── model_evaluator.py     # Evaluates models
│   └── model_comparison.py    # Compares models
└── pipeline/          # Orchestration
    └── training_pipeline.py    # End-to-end training pipeline
```

## How Modules Are Used

### 1. Data Source Module (`acm_forecast/data/data_source.py`)

**Purpose**: Loads Azure cost data from Databricks Delta tables

**Used in**: `TrainingPipeline.run()` - Step 1

**Example Usage**:
```python
from acm_forecast.data.data_source import DataSource

data_source = DataSource(spark)
df = data_source.load_from_delta(
    start_date="2023-01-01",
    end_date="2024-01-01",
    category="Compute"
)
```

**Key Methods**:
- `load_from_delta()`: Loads data from Delta table
- `get_data_profile()`: Gets data statistics
- `validate_data_availability()`: Checks data freshness
- `map_attributes()`: Maps source to target attributes

---

### 2. Data Quality Module (`acm_forecast/data/data_quality.py`)

**Purpose**: Validates data quality (completeness, accuracy, consistency, timeliness)

**Used in**: `TrainingPipeline.run()` - Step 2

**Example Usage**:
```python
from acm_forecast.data.data_quality import DataQualityValidator

quality_validator = DataQualityValidator(spark)
quality_results = quality_validator.comprehensive_validation(df)

# Check quality score
if quality_results["quality_score"] < 80:
    print("Warning: Low data quality!")
```

**Key Methods**:
- `validate_completeness()`: Checks for missing values
- `validate_accuracy()`: Checks for data errors
- `validate_consistency()`: Checks for duplicates
- `validate_timeliness()`: Checks data freshness
- `comprehensive_validation()`: Runs all validations

---

### 3. Data Preparation Module (`acm_forecast/data/data_preparation.py`)

**Purpose**: Prepares and cleans data for modeling

**Used in**: `TrainingPipeline.run()` - Steps 3, 4, 5

**Example Usage**:
```python
from acm_forecast.data.data_preparation import DataPreparation

data_prep = DataPreparation(spark)

# Aggregate daily costs
daily_df = data_prep.aggregate_daily_costs(df, group_by=["MeterCategory"])

# Handle missing values
daily_df = data_prep.handle_missing_values(daily_df)

# Split into train/validation/test
train_df, val_df, test_df = data_prep.split_time_series(daily_df)

# Prepare for Prophet
prophet_data = data_prep.prepare_for_prophet(train_df)

# Prepare for ARIMA
arima_data = data_prep.prepare_for_arima(train_df)
```

**Key Methods**:
- `aggregate_daily_costs()`: Aggregates to daily level
- `handle_missing_values()`: Fills missing data
- `detect_outliers()`: Identifies outliers
- `split_time_series()`: Splits chronologically
- `prepare_for_prophet()`: Formats for Prophet
- `prepare_for_arima()`: Formats for ARIMA
- `segment_by_category()`: Segments by category

---

### 4. Feature Engineering Module (`acm_forecast/data/feature_engineering.py`)

**Purpose**: Creates features for XGBoost model

**Used in**: `TrainingPipeline.run()` - Step 6 (XGBoost training)

**Example Usage**:
```python
from acm_forecast.data.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer(spark)

# Create all XGBoost features
xgboost_data = feature_engineer.prepare_xgboost_features(train_df)

# Or create features individually:
df_with_temporal = feature_engineer.create_temporal_features(df)
df_with_lags = feature_engineer.create_lag_features(df_with_temporal)
df_with_rolling = feature_engineer.create_rolling_features(df_with_lags)
df_with_derived = feature_engineer.create_derived_features(df_with_rolling)
```

**Key Methods**:
- `create_temporal_features()`: Year, month, day, cyclical encoding
- `create_lag_features()`: Lag 1, 2, 3, 7, 14, 30 days
- `create_rolling_features()`: Rolling mean, std, min, max
- `create_derived_features()`: Cost per unit, growth rates
- `prepare_xgboost_features()`: All features combined

---

### 5. Models Module (`acm_forecast/models/`)

**Purpose**: Implements forecasting models

**Used in**: `TrainingPipeline.run()` - Step 6

#### 5.1 Prophet Model (`acm_forecast/models/prophet_model.py`)

**Example Usage**:
```python
from acm_forecast.models.prophet_model import ProphetForecaster

prophet_model = ProphetForecaster(category="Total")
prophet_model.train(prophet_data)
forecast = prophet_model.predict(periods=30)
metrics = prophet_model.evaluate(forecast, actual_data)
```

**Key Methods**:
- `train()`: Trains Prophet model
- `predict()`: Generates forecasts
- `cross_validate()`: Cross-validation
- `evaluate()`: Calculates metrics
- `get_model_components()`: Extracts components

#### 5.2 ARIMA Model (`acm_forecast/models/arima_model.py`)

**Example Usage**:
```python
from acm_forecast.models.arima_model import ARIMAForecaster

arima_model = ARIMAForecaster(category="Total")
arima_model.train(arima_data)
forecast, conf_int = arima_model.predict(n_periods=30)
metrics = arima_model.evaluate(forecast, actual_data)
```

**Key Methods**:
- `test_stationarity()`: Tests if data is stationary
- `make_stationary()`: Applies differencing
- `train()`: Trains ARIMA model
- `predict()`: Generates forecasts
- `diagnose_residuals()`: Residual diagnostics
- `evaluate()`: Calculates metrics

#### 5.3 XGBoost Model (`acm_forecast/models/xgboost_model.py`)

**Example Usage**:
```python
from acm_forecast.models.xgboost_model import XGBoostForecaster

xgboost_model = XGBoostForecaster(category="Total")
xgboost_model.train(xgboost_data)
forecast = xgboost_model.predict(test_data)
metrics = xgboost_model.evaluate(forecast, actual_data)
importance = xgboost_model.get_feature_importance()
```

**Key Methods**:
- `prepare_features()`: Prepares features
- `train()`: Trains XGBoost model
- `predict()`: Generates forecasts
- `get_feature_importance()`: Feature importance
- `evaluate()`: Calculates metrics

---

### 6. Evaluation Module (`acm_forecast/evaluation/`)

**Purpose**: Evaluates and compares model performance

**Used in**: `TrainingPipeline.run()` - Step 7

#### 6.1 Performance Metrics (`acm_forecast/evaluation/performance_metrics.py`)

**Example Usage**:
```python
from acm_forecast.evaluation.performance_metrics import PerformanceMetrics

metrics_calc = PerformanceMetrics()
metrics = metrics_calc.calculate_metrics(y_true, y_pred)
# Returns: {'rmse': X, 'mae': Y, 'mape': Z, 'r2': W}
```

**Key Methods**:
- `calculate_metrics()`: RMSE, MAE, MAPE, R²
- `calculate_by_horizon()`: Metrics by forecast horizon
- `create_performance_summary()`: Summary table

#### 6.2 Model Evaluator (`acm_forecast/evaluation/model_evaluator.py`)

**Example Usage**:
```python
from acm_forecast.evaluation.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(forecast, actual, "Prophet")
sensitivity = evaluator.sensitivity_analysis(base_forecast, actual, perturbations)
benchmarks = evaluator.benchmark_comparison(model_forecast, actual, naive_forecasts)
```

**Key Methods**:
- `evaluate_model()`: Evaluates single model
- `sensitivity_analysis()`: Sensitivity testing
- `benchmark_comparison()`: Compare with naive methods

#### 6.3 Model Comparator (`acm_forecast/evaluation/model_comparison.py`)

**Example Usage**:
```python
from acm_forecast.evaluation.model_comparison import ModelComparator

comparator = ModelComparator()
comparison_df = comparator.compare_models(
    forecasts={"prophet": p_forecast, "arima": a_forecast, "xgboost": x_forecast},
    actual=actual_data
)
best_model = comparator.select_best_model(comparison_df, metric='mape')
ensemble = comparator.create_ensemble_forecast(forecasts, weights)
```

**Key Methods**:
- `compare_models()`: Compares multiple models
- `select_best_model()`: Selects best model
- `create_ensemble_forecast()`: Creates ensemble

---

## Complete Pipeline Flow

Here's how all modules work together in `TrainingPipeline.run()`:

```python
# Step 1: Load data (data_source)
df = self.data_source.load_from_delta(...)

# Step 2: Validate quality (data_quality)
quality = self.data_quality.comprehensive_validation(df)

# Step 3: Aggregate (data_preparation)
daily_df = self.data_prep.aggregate_daily_costs(df)

# Step 4: Prepare (data_preparation)
daily_df = self.data_prep.handle_missing_values(daily_df)

# Step 5: Split (data_preparation)
train_df, val_df, test_df = self.data_prep.split_time_series(daily_df)

# Step 6: Train models (models + feature_engineering)
# Prophet
prophet_data = self.data_prep.prepare_for_prophet(train_df)
self.prophet_forecaster.train(prophet_data)

# ARIMA
arima_data = self.data_prep.prepare_for_arima(train_df)
self.arima_forecaster.train(arima_data)

# XGBoost (uses feature_engineering)
xgboost_data = self.feature_engineer.prepare_xgboost_features(train_df)
self.xgboost_forecaster.train(xgboost_data)

# Step 7: Evaluate (evaluation)
metrics = self.evaluator.evaluate_model(forecast, actual)
comparison = self.comparator.compare_models(forecasts, actual)
```

## Direct Module Usage Examples

### Example 1: Standalone Data Quality Check

```python
from pyspark.sql import SparkSession
from acm_forecast.data.data_source import DataSource
from acm_forecast.data.data_quality import DataQualityValidator

spark = SparkSession.builder.appName("QualityCheck").getOrCreate()

# Load data
data_source = DataSource(spark)
df = data_source.load_from_delta()

# Check quality
quality_validator = DataQualityValidator(spark)
results = quality_validator.comprehensive_validation(df)
print(f"Quality Score: {results['quality_score']:.2f}%")
```

### Example 2: Standalone Feature Engineering

```python
import pandas as pd
from acm_forecast.data.feature_engineering import FeatureEngineer

# Load data
df = pd.read_csv("data/daily_costs.csv")

# Create features
feature_engineer = FeatureEngineer()
df_features = feature_engineer.prepare_xgboost_features(df)

print(f"Original columns: {len(df.columns)}")
print(f"Features created: {len(df_features.columns)}")
```

### Example 3: Standalone Model Training

```python
from acm_forecast.models.prophet_model import ProphetForecaster
from acm_forecast.data.data_preparation import DataPreparation
import pandas as pd

# Prepare data
data_prep = DataPreparation()
df = pd.read_csv("data/daily_costs.csv")
prophet_data = data_prep.prepare_for_prophet(df)

# Train model
model = ProphetForecaster(category="Total")
model.train(prophet_data)

# Generate forecast
forecast = model.predict(periods=30)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
```

## Summary

All modules are actively used in the pipeline:

✅ **data_source**: Loads data from Delta tables  
✅ **data_quality**: Validates data quality  
✅ **data_preparation**: Prepares and cleans data  
✅ **feature_engineering**: Creates ML features  
✅ **models**: Trains forecasting models  
✅ **evaluation**: Evaluates and compares models  

The `TrainingPipeline` orchestrates all these modules to provide an end-to-end solution.

