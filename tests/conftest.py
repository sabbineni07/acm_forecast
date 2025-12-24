"""
Pytest configuration and shared fixtures
"""
import pytest
import warnings
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
import os
import pandas as pd
from acm_forecast.config import AppConfig, DataConfig, ModelConfig, TrainingConfig, FeatureConfig
from acm_forecast.config import PerformanceConfig, RegistryConfig, MonitoringConfig, ForecastConfig
from acm_forecast.config import ProphetConfig, ArimaConfig, XGBoostConfig


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing"""
    config_data = {
        "name": "test_config",
        "data": {
            "delta_table_path": "test.database.test_table",
            "database_name": "test_database",
            "table_name": "test_table",
            "min_historical_months": 6,
            "max_data_delay_hours": 24,
        },
        "model": {
            "prophet": {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "seasonality_mode": "multiplicative",
                "changepoint_prior_scale": 0.05,
                "holidays_prior_scale": 10.0,
                "uncertainty_samples": 1000,
            },
            "arima": {
                "seasonal": True,
                "seasonal_period": 12,
                "max_p": 5,
                "max_d": 2,
                "max_q": 5,
                "information_criterion": "aic",
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "early_stopping_rounds": 10,
            },
        },
        "training": {
            "train_split": 0.7,
            "validation_split": 0.15,
            "test_split": 0.15,
            "min_training_months": 6,
        },
        "feature": {
            "target_column": "cost_in_billing_currency",
            "date_column": "usage_date",
            "lag_periods": [1, 7, 30],
            "rolling_windows": [7, 30],
        },
        "performance": {
            "target_mape": 10.0,
            "target_r2": 0.8,
            "warning_mape": 15.0,
            "critical_mape": 25.0,
        },
        "registry": {
            "mlflow_experiment_name": "test_experiment",
            "mlflow_tracking_uri": "file:///tmp/mlruns",
        },
        "monitoring": {
            "monthly_retraining": True,
            "realtime_monitoring": False,
        },
        "forecast": {
            "forecast_horizons_days": [30, 60, 90],
            "monthly_forecasts": True,
        },
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_app_config(temp_config_file):
    """Create a sample AppConfig instance from temp config file"""
    return AppConfig.from_yaml(temp_config_file)


@pytest.fixture
def sample_data_config():
    """Create a sample DataConfig instance"""
    return DataConfig(
        delta_table_path="test.database.test_table",
        database_name="test_database",
        table_name="test_table",
        min_historical_months=6,
        max_data_delay_hours=24,
    )


@pytest.fixture
def sample_model_config():
    """Create a sample ModelConfig instance"""
    prophet_config = ProphetConfig(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10.0,
        uncertainty_samples=1000,
    )
    arima_config = ArimaConfig(
        seasonal=True,
        seasonal_period=12,
        max_p=5,
        max_d=2,
        max_q=5,
        information_criterion="aic",
    )
    xgboost_config = XGBoostConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        early_stopping_rounds=10,
    )
    return ModelConfig(
        prophet=prophet_config,
        arima=arima_config,
        xgboost=xgboost_config,
    )


@pytest.fixture
def sample_training_config():
    """Create a sample TrainingConfig instance"""
    return TrainingConfig(
        train_split=0.7,
        validation_split=0.15,
        test_split=0.15,
        min_training_months=6,
    )


@pytest.fixture
def sample_feature_config():
    """Create a sample FeatureConfig instance"""
    return FeatureConfig(
        target_column="cost_in_billing_currency",
        date_column="usage_date",
        lag_periods=[1, 7, 30],
        rolling_windows=[7, 30],
    )


@pytest.fixture
def sample_performance_config():
    """Create a sample PerformanceConfig instance"""
    return PerformanceConfig(
        target_mape=10.0,
        target_r2=0.8,
        warning_mape=15.0,
        critical_mape=25.0,
    )


@pytest.fixture
def sample_registry_config():
    """Create a sample RegistryConfig instance"""
    return RegistryConfig(
        mlflow_experiment_name="test_experiment",
        mlflow_tracking_uri="file:///tmp/mlruns",
    )


@pytest.fixture
def sample_monitoring_config():
    """Create a sample MonitoringConfig instance"""
    return MonitoringConfig(
        monthly_retraining=True,
        realtime_monitoring=False,
    )


@pytest.fixture
def sample_forecast_config():
    """Create a sample ForecastConfig instance"""
    return ForecastConfig(
        forecast_horizons_days=[30, 60, 90],
        monthly_forecasts=True,
    )


@pytest.fixture
def mock_spark_session(monkeypatch):
    """Mock PySpark SparkSession for testing"""
    try:
        from unittest.mock import MagicMock
        spark = MagicMock()
        spark.sql.return_value = MagicMock()
        return spark
    except ImportError as ie:
        raise ie
        # pytest.skip("Mock not available")


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
@pytest.mark.filterwarnings("ignore:unclosed <socket.socket")
def spark_session():
    """Create a Spark session with Delta Lake support for testing
    
    Note: Requires Java 8 or 11 to be installed and JAVA_HOME set.
    Tests will be skipped if Java is not available.
    """
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip
        import os

        # Check for Java
        # java_home = os.environ.get("JAVA_HOME")
        # if not java_home:
        #     pytest.skip("JAVA_HOME not set. Java 8 or 11 is required for PySpark tests.")

        builder = SparkSession.builder \
            .appName("ACM_Forecast_Test") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.sql.warehouse.dir", "/tmp/test_spark_warehouse") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.ui.enabled", "false") \
            .config("spark.executor.memory", "1g") \
            .config("spark.executor.cores", "1")

        try:
            spark = configure_spark_with_delta_pip(builder).getOrCreate()
        except Exception as e:
            # if "JAVA_GATEWAY_EXITED" in str(e) or "Java" in str(e):
            #     pytest.skip(f"Java is required but not available: {e}")
            raise e
            # Fallback if delta-pip not available
            # spark = builder.getOrCreate()

        yield spark
        spark.sparkContext.stop()
        spark.stop()

        # # Suppress the ResourceWarning only during teardown
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed <socket.socket.*")
        #     spark.sparkContext.stop()
        #     spark.stop()
    # except ImportError as e:
    #     pytest.skip(f"PySpark or Delta Lake not available: {e}")
    except Exception as e:
        # if "JAVA_GATEWAY_EXITED" in str(e) or "Java" in str(e) or "java" in str(e).lower():
        #     pytest.skip(f"Java is required but not available: {e}")
        raise e


@pytest.fixture(scope="session")
def test_delta_table_path(tmp_path_factory, spark_session):
    """Generate test Delta table and return path"""
    import sys
    from pathlib import Path
    
    # Get test config path
    test_config_path = Path(__file__).parent / "config" / "test_config.yaml"
    
    # Generate sample data if needed
    delta_table_dir = tmp_path_factory.mktemp("test_delta_data")
    delta_table_path = str(delta_table_dir / "test_sample_costs")
    
    # Check if Delta table already exists
    delta_path_obj = Path(delta_table_path)
    if delta_path_obj.exists() and any(delta_path_obj.iterdir()):
        # Table exists, return path
        return delta_table_path
    
    # Generate sample data
    print("\nGenerating test Delta table...")
    try:
        # Import sample data generator
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.generate_sample_cost_data import generate_sample_data
        from datetime import datetime
        
        # Generate 90 days of sample data
        df = generate_sample_data(
            days=90,
            records_per_day=50,
            subscription_count=3,
            start_date=datetime(2024, 1, 1)
        )
        
        # Convert usage_date to pandas datetime
        df['usage_date'] = pd.to_datetime(df['usage_date'])
        
        # Convert DataFrame to list of dictionaries to avoid numpy dtype pickle issues
        # This ensures all types are Python native
        print(f"  Converting {len(df):,} records to Python dicts...")
        from datetime import datetime as dt
        rows = []
        for _, row in df.iterrows():
            # Convert pandas Timestamp to Python datetime
            usage_date_val = row['usage_date']
            if pd.isna(usage_date_val):
                usage_date_val = None
            elif isinstance(usage_date_val, pd.Timestamp):
                usage_date_val = usage_date_val.to_pydatetime()
            elif isinstance(usage_date_val, str):
                usage_date_val = pd.to_datetime(usage_date_val).to_pydatetime()
            
            rows.append({
                'usage_date': usage_date_val,
                'cost_in_billing_currency': float(row['cost_in_billing_currency']),
                'quantity': float(row['quantity']),
                'meter_category': str(row['meter_category']),
                'resource_location': str(row['resource_location']),
                'subscription_id': str(row['subscription_id']),
                'effective_price': float(row['effective_price']),
                'billing_currency_code': str(row['billing_currency_code']),
                'plan_name': str(row['plan_name']),
                'meter_sub_category': str(row['meter_sub_category']),
                'unit_of_measure': str(row['unit_of_measure']),
            })
        
        # Convert to Spark DataFrame with explicit schema
        print(f"  Creating Spark DataFrame...")
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
        from pyspark.sql.functions import to_date
        
        # Use TimestampType first (Spark handles pandas datetime as timestamp)
        schema = StructType([
            StructField("usage_date", TimestampType(), True),
            StructField("cost_in_billing_currency", DoubleType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("meter_category", StringType(), True),
            StructField("resource_location", StringType(), True),
            StructField("subscription_id", StringType(), True),
            StructField("effective_price", DoubleType(), True),
            StructField("billing_currency_code", StringType(), True),
            StructField("plan_name", StringType(), True),
            StructField("meter_sub_category", StringType(), True),
            StructField("unit_of_measure", StringType(), True),
        ])
        
        # Create Spark DataFrame from list of dicts
        spark_df = spark_session.createDataFrame(rows, schema=schema)
        
        # Convert timestamp to date type
        spark_df = spark_df.withColumn("usage_date", to_date(spark_df["usage_date"]))
        
        # Write as Delta table
        print(f"  Writing Delta table to: {delta_table_path}")
        spark_df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save(delta_table_path)
        
        # Create database and register table
        print("  Registering table in test_db...")
        spark_session.sql("CREATE DATABASE IF NOT EXISTS test_db")
        spark_session.sql(f"""
            CREATE TABLE IF NOT EXISTS test_db.test_sample_costs 
            USING DELTA 
            LOCATION '{delta_table_path}'
        """)
        
        print(f"✅ Test Delta table created with {len(df):,} records")
        return delta_table_path
        
    except Exception as e:
        print(f"⚠️  Could not generate Delta table: {e}")
        # Raise the error instead of skipping to catch real issues
        raise e
        # pytest.skip(f"Delta table generation failed: {e}")


@pytest.fixture
def test_app_config(tmp_path, test_delta_table_path):
    """Create AppConfig from test config with actual Delta table path"""
    import yaml
    from pathlib import Path
    from acm_forecast.config import AppConfig
    
    # Load test config
    test_config_path = Path(__file__).parent / "config" / "test_config.yaml"
    with open(test_config_path) as f:
        config_data = yaml.safe_load(f)
    
    # Update Delta table path to use generated table
    # The config uses test_db.test_sample_costs, which should work if table is registered
    # If not, we can override with full path
    
    # Create temporary config file
    temp_config_file = tmp_path / "test_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return AppConfig.from_yaml(str(temp_config_file))

