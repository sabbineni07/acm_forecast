# Configuration Guide

This directory contains YAML-based configuration for the Azure Cost Management Forecasting system.

## Quick Start

1. **Copy example config**:
   ```bash
   cp src/config/config.example.yaml src/config/config.yaml
   ```

2. **Edit config.yaml** with your settings

3. **Use in code**:
   ```python
   from src.config import AppConfig
   
   # Load configuration (defaults to src/config/config.yaml)
   config = AppConfig.from_yaml()
   
   # Access configs
   print(config.data.delta_table_path)
   print(config.model.prophet.yearly_seasonality)
   ```

## Configuration Files

- **`src/config/config.yaml`**: Main configuration file (create from example)
- **`src/config/config.example.yaml`**: Example configuration template

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
- Forecast horizons
- Granularity settings

## Usage Examples

### Basic Usage

```python
from src.config import AppConfig

# Load default config (from src/config/config.yaml)
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
from src.config import AppConfig
from src.pipeline.training_pipeline import TrainingPipeline

# Load config (defaults to src/config/config.yaml)
config = AppConfig.from_yaml()

# Use in pipeline
pipeline = TrainingPipeline(spark, config)
results = pipeline.run(category="Total")
```

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

1. **Never commit `src/config/config.yaml`** - Add to `.gitignore`
2. **Use `src/config/config.example.yaml`** as template
3. **Use environment variables** for sensitive data
4. **Validate config** before running pipelines
5. **Document custom configs** in your team

