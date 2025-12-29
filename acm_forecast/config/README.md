# Configuration Guide

This directory contains YAML-based configuration for the Azure Cost Management Forecasting system.

## Quick Start

1. **Copy example config**:
   ```bash
   cp acm_forecast/config/config.example.yaml acm_forecast/config/config.yaml
   ```

2. **Edit config.yaml** with your settings

3. **Use in code**:
   ```python
   from acm_forecast.config import AppConfig
   
   # Load configuration (defaults to acm_forecast/config/config.yaml)
   config = AppConfig.from_yaml()
   
   # Access configs
   print(config.data.delta_table_path)
   print(config.model.prophet.yearly_seasonality)
   ```

## Configuration Files

- **`acm_forecast/config/config.yaml`**: Main configuration file (create from example)
- **`acm_forecast/config/config.example.yaml`**: Example configuration template

## Configuration Structure

### Data Configuration
- Delta table paths
- Data constraints
- Regional distribution
- Cost categories

### Model Configuration
- Prophet model parameters
- ARIMA model parameters
- XGBoost model parameters

### Training Configuration
- Data splitting ratios
- Cross-validation settings
- Minimum data requirements

### Feature Engineering
- Lag periods
- Rolling windows
- Column names

### Performance Targets
- Target metrics (MAPE, R²)
- Alert thresholds

### Model Registry
- MLflow settings
- Model names
- Model stages

### Monitoring
- Retraining schedule
- Monitoring frequency

### Forecast
- Forecast horizons (supports multiple horizons)
- Granularity settings (daily, weekly, monthly)

### Plugins Configuration
- Plugin selection for each component type
- Plugin-specific configuration parameters
- Available plugins:
  - **data_source**: `acm` - ACM Delta data source
  - **data_quality**: `default` - Comprehensive validation
  - **data_preparation**: `acm` - ACM-specific preparation
  - **feature_engineer**: `default` - Temporal/lag/rolling features
  - **model**: `prophet`, `arima`, `xgboost` - Forecasting models
  - **forecaster**: `default` - Forecast generation
  - **model_registry**: `mlflow` - MLflow registry

## Usage Examples

### Basic Usage

```python
from acm_forecast.config import AppConfig

# Load default config (from acm_forecast/config/config.yaml)
config = AppConfig.from_yaml()

# Or specify custom path
config = AppConfig.from_yaml("path/to/custom_config.yaml")

# Access configurations (nested structure matches YAML)
print(config.data.delta_table_path)
print(config.model.prophet.yearly_seasonality)
print(config.model.arima.seasonal_period)
print(config.model.xgboost.n_estimators)
print(config.training.train_split)
```

### Using in Pipeline

```python
from acm_forecast.config import AppConfig
from acm_forecast.core import AppRunner
from acm_forecast.pipeline.training_pipeline import TrainingPipeline

# Load config
config = AppConfig.from_yaml("path/to/config.yaml")

# Option 1: Using AppRunner (recommended)
runner = AppRunner(config_path="path/to/config.yaml")
runner.run()

# Option 2: Using Pipeline classes directly
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ACM").getOrCreate()
pipeline = TrainingPipeline(config, spark)
results = pipeline.run(category="Total")
```

### Plugin Configuration Example

```python
# In YAML config file:
plugins:
  data_source:
    name: "acm"
    config: {}
  data_quality:
    name: "default"
    config:
      additional_completeness_columns: ["meter_category", "resource_location"]
      currency_column: "billing_currency_code"
      expected_currency: "USD"
  data_preparation:
    name: "acm"
    config: {}
  feature_engineer:
    name: "default"
    config:
      quantity_column: "quantity"
  model:
    name: "prophet"
    config: {}
```

See `config.example.yaml` for the complete configuration template with inline comments.

### Environment Variables

You can override config values with environment variables:

```bash
export DELTA_TABLE_PATH="custom.path.to.table"
export MLFLOW_TRACKING_URI="http://localhost:5000"
export ACM_CONFIG_PATH="/path/to/custom_config.yaml"
```

## Reloading Configuration

To reload configuration after changes, simply call `from_yaml()` again:

```python
config = AppConfig.from_yaml()
# ... make changes to config.yaml ...
config = AppConfig.from_yaml()  # Reload from file
```

## Validation

The configuration loader validates:
- Required fields
- Data types
- Value ranges
- File existence

## Best Practices

1. **Never commit `acm_forecast/config/config.yaml`** - Add to `.gitignore`
2. **Use `acm_forecast/config/config.example.yaml`** as template
3. **Use environment variables** for sensitive data
4. **Validate config** before running pipelines
5. **Document custom configs** in your team

