# Source Code Documentation

This directory contains the plugin-based implementation of the Azure Cost Management Forecasting Framework.

## Directory Structure

```
acm_forecast/
├── config/              # YAML-based configuration system
│   ├── config.example.yaml  # Configuration template
│   └── specifications.py    # Pydantic configuration classes
├── core/                 # Core framework components
│   ├── app_runner.py        # AppRunner - main orchestration class
│   ├── plugin_registry.py   # PluginFactory for plugin management
│   ├── base_plugin.py       # Base plugin class
│   └── interfaces.py        # Plugin interface definitions
├── plugins/              # Pluggable components
│   ├── data_source/         # Data source plugins (acm)
│   ├── data_quality/        # Data quality plugins (default)
│   ├── data_preparation/    # Data preparation plugins (acm)
│   ├── feature_engineer/    # Feature engineering plugins (default)
│   ├── models/              # Model plugins (prophet, arima, xgboost)
│   ├── forecasters/         # Forecaster plugins (default)
│   └── model_registry/      # Registry plugins (mlflow)
├── pipeline/             # Pipeline orchestration
│   ├── training_pipeline.py
│   └── forecast_pipeline.py
├── evaluation/           # Model evaluation and metrics
├── monitoring/           # Performance monitoring and retraining
└── examples/             # Example scripts
```

## Plugin Architecture

The framework uses a plugin-based architecture. All components are implemented as plugins:

- **Data Source Plugins**: Load data from various sources (e.g., `acm` for Delta tables)
- **Data Quality Plugins**: Validate data quality (e.g., `default` for comprehensive validation)
- **Data Preparation Plugins**: Prepare and transform data (e.g., `acm` for ACM-specific prep)
- **Feature Engineer Plugins**: Create features (e.g., `default` for temporal/lag/rolling features)
- **Model Plugins**: Forecasting models (`prophet`, `arima`, `xgboost`)
- **Forecaster Plugins**: Generate forecasts (e.g., `default`)
- **Model Registry Plugins**: Store and manage models (e.g., `mlflow`)

## Usage

### Using AppRunner (Recommended)

```python
from acm_forecast.core import AppRunner

# Initialize with configuration
runner = AppRunner(config_path="path/to/config.yaml")

# Run complete pipeline
runner.run()

# Generate forecasts only
runner.generate_forecasts(category="Total")
```

### Using Pipeline Classes

```python
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.pipeline import TrainingPipeline

# Load configuration
config = AppConfig.from_yaml("path/to/config.yaml")

# Initialize Spark
spark = SparkSession.builder.appName("ACM_Forecasting").getOrCreate()

# Create and run pipeline
pipeline = TrainingPipeline(config, spark)
results = pipeline.run(category="Total")
```

### Using PluginFactory Directly

```python
from acm_forecast.core import PluginFactory
from acm_forecast.config import AppConfig

config = AppConfig.from_yaml("path/to/config.yaml")
factory = PluginFactory()

# Create plugins
data_source = factory.create_data_source(config, spark, plugin_name="acm")
data_prep = factory.create_data_preparation(config, spark, plugin_name="acm")
model = factory.create_model(config, category="Total", plugin_name="prophet")
```

## Configuration

Configuration is managed through YAML files with Pydantic validation. See `config/README.md` for details.

Key configuration sections:
- `data`: Data source settings
- `model`: Model hyperparameters
- `training`: Training and validation settings
- `feature`: Feature engineering settings
- `performance`: Performance targets
- `registry`: MLflow registry settings
- `monitoring`: Monitoring and retraining settings
- `forecast`: Forecast generation settings
- `plugins`: Plugin configuration

## Documentation

- **Model Documentation**: See `MODEL_DOCUMENTATION.md` for comprehensive model details
- **Examples**: See `examples/README_E2E.md` for usage examples
- **Configuration**: See `config/README.md` for configuration guide


