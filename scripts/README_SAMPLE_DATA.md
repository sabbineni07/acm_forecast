# Sample Data Generation

## Overview

Sample data generation is now integrated directly into the `DataSource` class in `acm_forecast/data/data_source.py`. You can generate synthetic Azure Cost Management data either through configuration or by calling the method directly.

## Quick Start

### Option 1: Configuration-Driven (Recommended)

Enable sample data generation in your YAML configuration:

```yaml
data:
  # ... other data config ...
  generate_sample_data: true
  sample_data_days: 365
  sample_data_records_per_day: 100
  sample_data_subscriptions: 3
  sample_data_start_date: "2024-01-01"  # Optional, defaults to days ago from today
```

Then use `DataSource` normally - it will generate data automatically:

```python
from acm_forecast.data.data_source import DataSource
from acm_forecast.config import AppConfig
from pyspark.sql import SparkSession

config = AppConfig.from_yaml()
spark = SparkSession.builder.appName("Test").getOrCreate()

data_source = DataSource(config, spark)
# Automatically generates sample data if generate_sample_data is true
df = data_source.load_from_delta()
```

### Option 2: Direct Method Call

Generate data directly without modifying config:

```python
from acm_forecast.data.data_source import DataSource
from acm_forecast.config import AppConfig

config = AppConfig.from_yaml()
# Temporarily enable generation
config.data.generate_sample_data = True
config.data.sample_data_days = 365
config.data.sample_data_records_per_day = 100

data_source = DataSource(config, spark)
df = data_source.generate_sample_data()
```

### Option 3: Save to File Format

To save generated data to CSV, Parquet, or Delta:

```python
from acm_forecast.data.data_source import DataSource
from acm_forecast.config import AppConfig

config = AppConfig.from_yaml()
config.data.generate_sample_data = True
config.data.sample_data_days = 365

data_source = DataSource(config, spark)
df = data_source.generate_sample_data()

# Save as CSV
df.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/sample_costs.csv")

# Save as Parquet
df.coalesce(1).write.mode("overwrite").parquet("data/sample_costs.parquet")

# Save as Delta table
df.write.format("delta").mode("overwrite").save("data/delta/sample_costs")

# Register as table
spark.sql("CREATE DATABASE IF NOT EXISTS test_db")
spark.sql("""
    CREATE TABLE IF NOT EXISTS test_db.sample_costs 
    USING DELTA 
    LOCATION 'data/delta/sample_costs'
""")
```

## Data Schema

The generated data includes all required and recommended columns:

### Required Columns
- `usage_date` (DATE) - Time series index
- `cost_in_billing_currency` (DECIMAL) - Target variable for forecasting
- `quantity` (DECIMAL) - Usage quantity
- `meter_category` (STRING) - Cost category (Compute, Storage, Network, etc.)
- `resource_location` (STRING) - Azure region

### Recommended Columns
- `subscription_id` (STRING) - Azure subscription ID
- `effective_price` (DECIMAL) - Price per unit
- `billing_currency_code` (STRING) - Currency code (default: USD)
- `plan_name` (STRING) - Service plan/tier name

### Optional Columns
- `meter_sub_category` (STRING) - Subcategory within meter category
- `unit_of_measure` (STRING) - Unit of measurement

## Configuration Parameters

All parameters are configurable through `DataConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_sample_data` | bool | `false` | Enable sample data generation |
| `sample_data_days` | int | `365` | Number of days of historical data |
| `sample_data_records_per_day` | int | `100` | Average number of records per day |
| `sample_data_subscriptions` | int | `3` | Number of different subscriptions |
| `sample_data_start_date` | str | `null` | Start date (YYYY-MM-DD, defaults to days ago) |

## Example: Generate and Use in Pipeline

```python
from acm_forecast.config import AppConfig
from acm_forecast.data.data_source import DataSource
from acm_forecast.data.data_preparation import DataPreparation
from acm_forecast.data.data_quality import DataQualityValidator

# Load config with sample data generation enabled
config = AppConfig.from_yaml("config_with_sample_data.yaml")

# Initialize components
data_source = DataSource(config, spark)
data_prep = DataPreparation(config, spark)
data_quality = DataQualityValidator(config, spark)

# Load data (generates sample data if enabled)
df = data_source.load_from_delta()

# Validate data quality
quality_results = data_quality.comprehensive_validation(df)
print(f"Data quality score: {quality_results['quality_score']:.2f}%")

# Prepare data for modeling
daily_df = data_prep.aggregate_daily_costs(df)
```

## Notes

- **All data is generated as PySpark DataFrames** - no pandas dependency
- **Native Python types only** - ensures compatibility with PySpark serialization
- **Realistic patterns** - includes trends, seasonality, and realistic cost distributions
- **Configurable** - all aspects can be controlled through YAML configuration

## Migration from Standalone Scripts

The standalone scripts (`generate_sample_cost_data.py`, `create_test_delta_table.py`) have been removed. All functionality is now available through:

1. **`DataSource.generate_sample_data()`** - Direct method call
2. **`DataSource.load_from_delta()`** - Configuration-driven generation
3. **YAML configuration** - Enable via `data.generate_sample_data: true`

This provides a unified, consistent API for both loading real data and generating sample data.
