# Azure Cost Management Forecasting - Source Code

This directory contains the modular implementation of the Azure Cost Management Forecasting Model, organized according to the MODEL_DOCUMENTATION.md structure.

## Directory Structure

```
src/
├── config/           # Configuration settings (Section 7)
│   └── settings.py   # All configuration classes
├── data/             # Data module (Section 3)
│   ├── data_source.py          # Data sourcing (3.1)
│   ├── data_preparation.py     # Data preparation (3.3)
│   ├── data_quality.py         # Data quality validation (3.1.4)
│   └── feature_engineering.py  # Feature engineering (3.3.4, 5.1)
├── models/           # Model implementations (Section 4-5)
│   ├── prophet_model.py   # Prophet model (4.2.1, 5.2.1)
│   ├── arima_model.py     # ARIMA model (4.2.1, 5.2.1)
│   └── xgboost_model.py   # XGBoost model (4.2.1, 5.2.1)
├── evaluation/       # Model evaluation (Section 6)
│   ├── performance_metrics.py  # Performance metrics (6.1)
│   ├── model_evaluator.py     # Model evaluation (6.1, 6.2)
│   └── model_comparison.py    # Model comparison (6.1, 5.2.2)
├── registry/         # Model registry (Section 7.2-7.3)
│   ├── model_registry.py      # MLflow integration (7.2)
│   └── model_versioning.py    # Versioning and controls (7.3)
├── monitoring/       # Monitoring (Section 8)
│   ├── performance_monitor.py  # Performance monitoring (8.1)
│   ├── data_drift_monitor.py   # Data drift detection (8.2)
│   └── retraining_scheduler.py # Retraining schedule (8.3)
└── pipeline/         # Pipeline orchestration (Section 7.1)
    ├── training_pipeline.py    # Training pipeline
    └── forecast_pipeline.py   # Forecast pipeline
```

## Usage

### Training Pipeline

```python
from pyspark.sql import SparkSession
from src.pipeline.training_pipeline import TrainingPipeline

# Initialize Spark
spark = SparkSession.builder.appName("ACM_Forecasting").getOrCreate()

# Create pipeline
pipeline = TrainingPipeline(spark)

# Run training
results = pipeline.run(category="Compute", start_date="2023-01-01", end_date="2024-01-01")
```

### Forecast Generation

```python
from src.pipeline.forecast_pipeline import ForecastPipeline

# Create forecast pipeline
forecast_pipeline = ForecastPipeline(spark)

# Generate forecasts
forecasts = forecast_pipeline.generate_forecasts(category="Compute", horizons=[30, 90, 180])
```

### Model Registry

```python
from src.registry.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register model
version = registry.register_model(
    model=trained_model,
    model_name="azure_cost_forecast_prophet",
    model_type="prophet",
    metrics={"mape": 8.5, "r2": 0.85},
    category="Compute"
)

# Promote to production
registry.promote_model("azure_cost_forecast_prophet", version, "Production")
```

## Configuration

All configuration is managed through `src/config/settings.py`. Key configuration classes:

- `DataConfig`: Data source and processing settings
- `ModelConfig`: Model hyperparameters
- `TrainingConfig`: Training and validation settings
- `FeatureConfig`: Feature engineering settings
- `PerformanceConfig`: Performance targets and thresholds
- `RegistryConfig`: MLflow registry settings
- `MonitoringConfig`: Monitoring and retraining settings
- `ForecastConfig`: Forecast generation settings

## Documentation

See `MODEL_DOCUMENTATION.md` in the project root for complete documentation of all sections.


