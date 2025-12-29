# Azure Cost Management Forecasting Framework

A production-ready, plugin-based forecasting framework for Azure cost management using Prophet, ARIMA, and XGBoost models with PySpark-based data processing.

## 🏗️ Project Structure

```
acm_forecast/
├── acm_forecast/               # Main source code package
│   ├── config/                 # YAML-based configuration system
│   │   ├── config.example.yaml # Example configuration template
│   │   └── specifications.py   # Pydantic configuration classes
│   ├── core/                   # Core framework components
│   │   ├── app_runner.py       # AppRunner - main orchestration class
│   │   ├── plugin_registry.py  # PluginFactory for dynamic plugin loading
│   │   ├── base_plugin.py      # Base plugin interface
│   │   └── interfaces.py       # Plugin interface definitions
│   ├── plugins/                # Pluggable components (plugin architecture)
│   │   ├── data_source/        # Data source plugins (acm)
│   │   ├── data_quality/       # Data quality plugins (default)
│   │   ├── data_preparation/   # Data preparation plugins (acm)
│   │   ├── feature_engineer/   # Feature engineering plugins (default)
│   │   ├── models/             # Model plugins (prophet, arima, xgboost)
│   │   ├── forecasters/        # Forecaster plugins (default)
│   │   └── model_registry/     # Registry plugins (mlflow)
│   ├── pipeline/               # Pipeline orchestration
│   │   ├── training_pipeline.py
│   │   └── forecast_pipeline.py
│   ├── evaluation/             # Model evaluation and metrics
│   ├── monitoring/             # Performance monitoring and retraining
│   └── examples/               # Example scripts
│       ├── run_end_to_end.py   # Complete end-to-end example
│       ├── run_training.py     # Training pipeline example
│       └── run_forecast.py     # Forecast generation example
├── tests/                      # Test suite (unit, integration, e2e)
├── pyproject.toml              # Package configuration
├── Makefile                    # Build and development commands
└── requirements.txt            # Python dependencies
```

## 🚀 Features

- **Plugin-Based Architecture**: Flexible, extensible plugin system for all components
- **Multiple Forecasting Models**: Prophet, ARIMA, and XGBoost
- **Production-Ready**: MLflow integration, monitoring, and automated retraining
- **PySpark-Based**: Scalable distributed processing for large datasets
- **YAML Configuration**: Simple, maintainable configuration management
- **AppRunner**: Simple high-level interface for running pipelines

## 🚀 Quick Start

### Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e ".[dev]"
```

### Using Makefile

```bash
# Build wheel package
make build

# Run tests
make test

# Run specific test types
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only

# Code quality
make format            # Format code with black
make lint              # Lint code with flake8
make check             # Run linting and tests

# Clean build artifacts
make clean-build
```

See `make help` for all available commands.

### Basic Usage with AppRunner

```python
from acm_forecast.core import AppRunner

# Initialize with configuration file
runner = AppRunner(config_path="acm_forecast/examples/config_end_to_end.yaml")

# Run complete pipeline (load data, train, forecast, evaluate)
runner.run()

# Run specific steps
runner.run(steps=['load_data', 'prepare_data', 'train_model', 'forecast'])

# Generate forecasts only
runner.generate_forecasts(category="Total")
```

### Using Pipeline Classes Directly

```python
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.pipeline import TrainingPipeline

# Load configuration
config = AppConfig.from_yaml("path/to/config.yaml")

# Initialize Spark
spark = SparkSession.builder.appName("ACM_Forecasting").getOrCreate()

# Create and run training pipeline
pipeline = TrainingPipeline(config, spark)
results = pipeline.run(category="Total")
```

### End-to-End Example

```bash
# Run complete end-to-end example
python -m acm_forecast.examples.run_end_to_end --config acm_forecast/examples/config_end_to_end.yaml
```

## 📖 Documentation

- **[Installation Guide](INSTALLATION.md)** - Setup and installation instructions
- **[Configuration Guide](acm_forecast/config/README.md)** - YAML configuration reference
- **[Model Documentation](acm_forecast/MODEL_DOCUMENTATION.md)** - Comprehensive model documentation
- **[Examples Guide](acm_forecast/examples/README_E2E.md)** - End-to-end examples and usage
- **[Testing Guide](tests/README.md)** - Test suite documentation
- **[Packaging Guide](PACKAGING.md)** - Building and distributing the package

## 🔌 Plugin Architecture

The framework uses a plugin-based architecture for extensibility. All components (data sources, models, forecasters, etc.) are implemented as plugins.

### Built-in Plugins

- **Data Source**: `acm` - Azure Cost Management Delta data source
- **Data Quality**: `default` - Comprehensive data quality validation
- **Data Preparation**: `acm` - ACM-specific data preparation
- **Feature Engineer**: `default` - Temporal, lag, and rolling features
- **Models**: `prophet`, `arima`, `xgboost` - Forecasting models
- **Forecaster**: `default` - Forecast generation
- **Model Registry**: `mlflow` - MLflow model registry

### Plugin Configuration

Plugins are configured in YAML:

```yaml
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
    name: "prophet"  # Uses model.selected_model if not specified
    config: {}
  forecaster:
    name: "default"
    config: {}
  model_registry:
    name: "mlflow"
    config: {}
```

See [Configuration Guide](acm_forecast/config/README.md) for details.

## ⚙️ Configuration

Configuration is managed through YAML files with Pydantic validation. Example configuration:

```yaml
name: "acm_forecast"
data:
  delta_table_path: "azure_cost_management.amortized_costs"
  database_name: "azure_cost_management"
  table_name: "amortized_costs"
model:
  selected_model: "prophet"
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
training:
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15
forecast:
  forecast_horizons_days: [30, 60, 90]
```

See `acm_forecast/config/config.example.yaml` for the complete configuration template.

## 📊 Model Comparison

The framework provides comprehensive comparison metrics:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
pytest tests/unit -m unit
pytest tests/integration -m integration
pytest tests/e2e -m e2e
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## 📦 Building and Distribution

```bash
# Build wheel package
make build

# Install locally
pip install dist/acm_forecast-1.0.0-py3-none-any.whl

# Install in development mode
pip install -e ".[dev]"
```

See [PACKAGING.md](PACKAGING.md) for detailed packaging instructions.

## 🐳 Docker Support

The project includes Docker configurations for development and production:

```bash
# Build Docker images
make docker-build

# Start services
make docker-up

# Run tests in Docker
make docker-test

# Run pipeline in Docker
make docker-run-pipeline
```

See `Makefile` help (`make help`) for all Docker commands.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run linting: `make lint`
6. Submit a pull request

## 📝 License

MIT License

## 🔗 Related Resources

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [MLflow Documentation](https://www.mlflow.org/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
