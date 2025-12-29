# End-to-End Forecasting Example

This example demonstrates the complete forecasting workflow using the plugin architecture.

## Overview

The end-to-end example (`run_end_to_end.py`) walks through all stages of the forecasting pipeline:

1. **Data Loading/Generation**: Load data from Delta table or generate sample data
2. **Data Preparation**: Aggregate, clean, and prepare data for modeling
3. **Data Quality Validation**: Run comprehensive data quality checks
4. **Model Training**: Train a forecasting model (Prophet, ARIMA, or XGBoost)
5. **Forecast Generation**: Generate forecasts for the specified horizon
6. **Visualization**: Create visualizations of historical data and forecasts

## Configuration

The example uses `config_end_to_end.yaml` which includes:

- **Data Configuration**: Delta table path, sample data generation settings
- **Model Configuration**: Model selection (prophet/arima/xgboost) and hyperparameters
- **Plugin Configuration**: Plugin selection for each component
- **Forecast Configuration**: Forecast horizon and confidence intervals

### Plugin Configuration

The YAML config includes a `plugins` section that specifies which plugin to use for each component:

```yaml
plugins:
  data_source:
    name: "acm"
    config: {}
  data_quality:
    name: "default"
    config: {}
  data_preparation:
    name: "acm"
    config: {}
  feature_engineer:
    name: "default"
    config: {}
  model:
    name: "prophet"  # Will use model.selected_model if not specified
    config: {}
  forecaster:
    name: "default"
    config: {}
  model_registry:
    name: "mlflow"
    config: {}
```

## Usage

### Basic Usage

```bash
# Run with default configuration
python -m acm_forecast.examples.run_end_to_end
```

### With Custom Configuration

```bash
# Use a custom config file
python -m acm_forecast.examples.run_end_to_end --config path/to/config.yaml
```

### Command Line Options

```bash
python -m acm_forecast.examples.run_end_to_end \
  --config config_end_to_end.yaml \
  --category "Total" \
  --output forecast_results.png
```

Options:
- `--config`: Path to YAML configuration file (default: `config_end_to_end.yaml`)
- `--category`: Cost category to forecast (default: "Total")
- `--output`: Output path for forecast visualization (default: `forecast_visualization.png`)

## Requirements

- Python 3.9+
- PySpark (with Java installed)
- Delta Lake
- Required Python packages: pandas, matplotlib, seaborn
- Model-specific packages (prophet, statsmodels, xgboost) depending on selected model

## Workflow Details

### Step 1: Data Loading/Generation

The script attempts to load data from the Delta table specified in the config. If the table doesn't exist or is empty, it generates sample data using the `ACMDeltaDataSource` plugin.

### Step 2: Data Preparation

The `ACMDataPreparation` plugin (registered as 'acm'):
- Aggregates data to daily level
- Splits data into training, validation, and test sets
- Prepares data in model-specific formats (Prophet, ARIMA, XGBoost)

### Step 3: Data Quality Validation

The `DefaultDataQuality` plugin runs comprehensive validation:
- Completeness checks
- Accuracy checks
- Consistency checks
- Timeliness checks

### Step 4: Model Training

Creates the selected model plugin and trains it on the prepared data:
- **Prophet**: Uses `ProphetModelPlugin`
- **ARIMA**: Uses `ARIMAModelPlugin`
- **XGBoost**: Uses `XGBoostModelPlugin`

### Step 5: Forecast Generation

Uses the trained model to generate forecasts for the specified horizon (default: 90 days).

### Step 6: Visualization

Creates a visualization showing:
- Historical data (blue line)
- Forecast (red dashed line)
- Confidence intervals (if available, red shaded area)

The visualization is saved to the specified output path.

## Example Output

The script produces:
1. Console logs for each step
2. A PNG visualization file with the forecast results
3. Validation results from data quality checks

## Customization

### Using Different Plugins

To use custom plugins, update the `plugins` section in the YAML config:

```yaml
plugins:
  data_source:
    name: "custom_data_source"
    config:
      custom_param: "value"
```

### Changing Model Parameters

Update the model-specific sections in the YAML:

```yaml
model:
  selected_model: "prophet"
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
    changepoint_prior_scale: 0.1  # Adjust as needed
```

### Adjusting Forecast Horizon

```yaml
forecast:
  forecast_horizon_days: 180  # 6 months
  confidence_interval: 0.95
```

## Troubleshooting

### Java Not Found

If you get Java-related errors, ensure Java is installed and `JAVA_HOME` is set:

```bash
export JAVA_HOME=$(/usr/libexec/java_home)
```

### Delta Table Errors

If Delta table operations fail, ensure:
- Delta Lake is properly configured
- Spark session has Delta extensions enabled
- Table path is accessible

### Model Training Errors

- Ensure model-specific dependencies are installed (prophet, statsmodels, xgboost)
- Check that data has sufficient historical records (at least 6 months recommended)
- Verify data quality validation passes

### Visualization Errors

- Ensure matplotlib and seaborn are installed
- Check that forecast DataFrame has expected columns (ds/yhat for Prophet, date/forecast for others)

## Next Steps

After running the end-to-end example:
1. Review the forecast visualization
2. Evaluate model performance using test data
3. Adjust model parameters based on results
4. Consider registering the model with MLflow for tracking
5. Set up automated retraining schedules

