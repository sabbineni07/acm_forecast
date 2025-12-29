"""
Pytest configuration and shared fixtures
"""
import pytest
import tempfile
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os
import yaml
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

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    yield spark
    spark.sparkContext.stop()
    spark.stop()


@pytest.fixture(scope="session")
def test_delta_table_path(tmp_path_factory, spark_session):
    """Generate test Delta table and return path"""
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
    
    # Generate sample data using PluginFactory
    print("\nGenerating test Delta table...")
    try:
        from acm_forecast.core import PluginFactory
        from acm_forecast.config import AppConfig
        
        # Load test config
        test_config_path = Path(__file__).parent / "config" / "test_config.yaml"
        test_config = AppConfig.from_yaml(str(test_config_path))
        
        # Enable sample data generation
        test_config.data.generate_sample_data = True
        test_config.data.sample_data_days = 90
        test_config.data.sample_data_records_per_day = 50
        test_config.data.sample_data_subscriptions = 3
        test_config.data.sample_data_start_date = "2024-01-01"
        
        # Create DataSource plugin and generate sample data
        factory = PluginFactory()
        data_source = factory.create_data_source(test_config, spark_session, plugin_name="acm")
        print(f"  Generating sample data using ACMDeltaDataSource plugin...")
        spark_df = data_source.generate_sample_data()
        print(f"  Generated sample data")
        
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
        
        # Count records after writing
        record_count = spark_session.read.format("delta").load(delta_table_path).count()
        print(f"✅ Test Delta table created with {record_count:,} records")
        return delta_table_path
        
    except Exception as e:
        print(f"⚠️  Could not generate Delta table: {e}")
        raise e


@pytest.fixture
def test_app_config(tmp_path, test_delta_table_path):
    """Create AppConfig from test config with actual Delta table path"""
    # Load test config
    test_config_path = Path(__file__).parent / "config" / "test_config.yaml"
    with open(test_config_path) as f:
        config_data = yaml.safe_load(f)

    # Create temporary config file
    temp_config_file = tmp_path / "test_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return AppConfig.from_yaml(str(temp_config_file))

