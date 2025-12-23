# Azure Cost Management Forecasting Project

This project implements multiple forecasting models for Azure cost management using a modular, production-ready architecture built with PySpark.

## 🏗️ Project Structure

```
acm_forecast/
├── acm_forecast/            # Main source code (package)
│   ├── config/              # Configuration settings
│   ├── data/                # Data processing modules
│   ├── models/              # Forecasting models (Prophet, ARIMA, XGBoost)
│   ├── evaluation/          # Model evaluation and comparison
│   ├── registry/            # Model registry (MLflow integration)
│   ├── monitoring/          # Model monitoring and retraining
│   ├── pipeline/            # Training and forecast pipelines
│   └── examples/            # Example scripts
│       ├── run_training.py          # Training pipeline script
│       ├── run_forecast.py          # Forecast generation script
│       └── run_complete_pipeline.py # End-to-end pipeline script
├── pyproject.toml           # Package configuration
└── requirements.txt         # Python dependencies
```

## 🚀 Features

- **Multiple Forecasting Models**:
  - Prophet (Facebook's time series forecasting)
  - ARIMA (AutoRegressive Integrated Moving Average)
  - XGBoost (Gradient Boosting)
- **Production-Ready Architecture**: Modular design with clear separation of concerns
- **Model Registry**: MLflow integration for model versioning and deployment
- **Monitoring**: Performance monitoring, data drift detection, and automated retraining
- **PySpark-Based**: Scalable distributed processing for large datasets

## 🚀 Getting Started

### Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

```bash
pip install -r requirements.txt
```

### Quick Start

#### Training Models

```bash
python acm_forecast/examples/run_training.py
```

#### Generating Forecasts

```bash
python acm_forecast/examples/run_forecast.py
```

#### Complete Pipeline

```bash
python acm_forecast/examples/run_complete_pipeline.py
```

## Usage

### Training Pipeline

```python
from pyspark.sql import SparkSession
from acm_forecast.pipeline.training_pipeline import TrainingPipeline

# Initialize Spark
spark = SparkSession.builder.appName("ACM_Forecasting").getOrCreate()

# Create pipeline
pipeline = TrainingPipeline(spark)

# Run training
results = pipeline.run(category="Compute", start_date="2023-01-01", end_date="2024-01-01")
```

### Forecast Generation

```python
from acm_forecast.pipeline.forecast_pipeline import ForecastPipeline

# Create forecast pipeline
forecast_pipeline = ForecastPipeline(spark)

# Generate forecasts
forecasts = forecast_pipeline.generate_forecasts(category="Compute", horizons=[30, 90, 180])
```

## Documentation

- [Source Code Documentation](acm_forecast/README.md) - Detailed source code documentation
- [Model Documentation](MODEL_DOCUMENTATION.md) - Complete model documentation
- [Module Usage Guide](MODULE_USAGE_GUIDE.md) - How to use each module
- [Run Forecast Guide](acm_forecast/examples/RUN_FORECAST_GUIDE.md) - Guide for generating forecasts
- [Installation Guide](INSTALLATION.md) - Installation and setup instructions

## Configuration

All configuration is managed through `acm_forecast/config/specifications.py`. See [acm_forecast/README.md](acm_forecast/README.md) for details.

## Model Comparison

The project provides comprehensive comparison metrics including:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Visualization of predictions vs actual costs
