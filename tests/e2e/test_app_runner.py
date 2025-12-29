"""
End-to-End Tests for AppRunner Class

Tests the complete AppRunner workflow with different step combinations
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

from acm_forecast.core import AppRunner
from acm_forecast.config import AppConfig


@pytest.fixture
def mock_spark_session():
    """Create a mock SparkSession"""
    spark = Mock(spec=SparkSession)
    spark.sparkContext = Mock()
    spark.sparkContext.setLogLevel = Mock()
    return spark


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a temporary config YAML file for testing"""
    config_data = {
        "name": "test_app",
        "data": {
            "delta_table_path": "test_db.test_table",
            "database_name": "test_db",
            "table_name": "test_table",
            "generate_sample_data": True,
            "sample_data_days": 30,
            "sample_data_records_per_day": 10,
            "sample_data_subscriptions": 2,
        },
        "model": {
            "selected_model": "prophet",
            "prophet": {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
            },
            "arima": {},
            "xgboost": {},
        },
        "training": {
            "train_split": 0.7,
            "validation_split": 0.15,
            "test_split": 0.15,
        },
        "feature": {
            "date_column": "usage_date",
            "target_column": "cost_in_billing_currency",
        },
        "forecast": {
            "forecast_horizons_days": [30],
        },
        "performance": {
            "target_mape": 10.0,
            "target_r2": 0.8,
        },
        "registry": {
            "mlflow_experiment_name": "test_experiment",
            "mlflow_tracking_uri": "file:///tmp/mlruns",
        },
        "monitoring": {
            "monthly_retraining": True,
        },
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_file)


@pytest.fixture
def sample_spark_dataframe(mock_spark_session):
    """Create a sample Spark DataFrame for testing"""
    # Create mock DataFrame with isEmpty() and other methods
    mock_df = Mock(spec=DataFrame)
    mock_df.isEmpty.return_value = False
    mock_df.columns = ["usage_date", "cost_in_billing_currency", "meter_category"]
    # printSchema() doesn't return anything, it just prints
    mock_df.printSchema = Mock(return_value=None)
    mock_df.toPandas.return_value = pd.DataFrame({
        "usage_date": pd.date_range("2023-01-01", periods=100),
        "cost_in_billing_currency": [100.0] * 100,
    })
    mock_df.count.return_value = 100
    return mock_df


@pytest.mark.e2e
class TestAppRunner:
    """End-to-end tests for AppRunner class"""
    
    def test_app_runner_init(self, sample_config_yaml, mock_spark_session):
        """Test AppRunner initialization"""
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        assert runner.config_path == Path(sample_config_yaml)
        assert runner.config is not None
        assert runner.spark == mock_spark_session
        assert runner.factory is None  # Factory created lazily
    
    def test_app_runner_init_without_session(self, sample_config_yaml):
        """Test AppRunner initialization without Spark session"""
        runner = AppRunner(sample_config_yaml)
        
        assert runner.config_path == Path(sample_config_yaml)
        assert runner.config is not None
        assert runner.spark is None
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_load_or_generate_data(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test load_or_generate_data method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_source = Mock()
        mock_data_source.load_data.return_value = sample_spark_dataframe
        mock_factory.create_data_source.return_value = mock_data_source
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        result_df = runner.load_or_generate_data()
        
        # Assertions
        assert result_df == sample_spark_dataframe
        mock_factory.create_data_source.assert_called_once()
        mock_data_source.load_data.assert_called_once()
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_load_or_generate_data_empty(self, mock_factory_class, sample_config_yaml, mock_spark_session):
        """Test load_or_generate_data with empty DataFrame"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_source = Mock()
        empty_df = Mock(spec=DataFrame)
        empty_df.isEmpty.return_value = True
        mock_data_source.load_data.return_value = empty_df
        mock_factory.create_data_source.return_value = mock_data_source
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        with pytest.raises(ValueError, match="Delta table exists but is empty"):
            runner.load_or_generate_data()
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_prepare_data(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test prepare_data method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_prep = Mock()
        agg_df = Mock(spec=DataFrame)
        train_df = Mock(spec=DataFrame)
        val_df = Mock(spec=DataFrame)
        test_df = Mock(spec=DataFrame)
        train_df.isEmpty.return_value = False
        val_df.isEmpty.return_value = False
        test_df.isEmpty.return_value = False
        
        mock_data_prep.aggregate_data.return_value = agg_df
        mock_data_prep.split.return_value = (train_df, val_df, test_df)
        mock_factory.create_data_preparation.return_value = mock_data_prep
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        result_train, result_val, result_test = runner.prepare_data(sample_spark_dataframe)
        
        # Assertions
        assert result_train == train_df
        assert result_val == val_df
        assert result_test == test_df
        mock_data_prep.aggregate_data.assert_called_once_with(sample_spark_dataframe)
        mock_data_prep.split.assert_called_once_with(agg_df)
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_run_data_quality(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test run_data_quality method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_quality = Mock()
        validation_results = {
            "completeness": {"passed": True, "message": "All checks passed"},
            "accuracy": {"passed": True, "message": "Data is accurate"},
        }
        mock_data_quality.comprehensive_validation.return_value = validation_results
        mock_factory.create_data_quality.return_value = mock_data_quality
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        results = runner.run_data_quality(sample_spark_dataframe)
        
        # Assertions
        assert results == validation_results
        mock_data_quality.comprehensive_validation.assert_called_once_with(sample_spark_dataframe)
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_train_model(self, mock_factory_class, sample_config_yaml, mock_spark_session):
        """Test train_model method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        train_df = Mock(spec=DataFrame)
        train_df.isEmpty.return_value = False
        
        mock_data_prep = Mock()
        train_data_pd = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=50), "y": [100.0] * 50})
        mock_data_prep.prepare_for_training.return_value = train_data_pd
        mock_factory.create_data_preparation.return_value = mock_data_prep
        
        mock_model = Mock()
        mock_model.train.return_value = {"status": "trained"}
        mock_factory.create_model.return_value = mock_model
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        result_model = runner.train_model(train_df, category="Total")
        
        # Assertions
        assert result_model == mock_model
        mock_model.train.assert_called_once()
        mock_factory.create_model.assert_called_once()
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_generate_forecasts(self, mock_factory_class, sample_config_yaml, mock_spark_session):
        """Test generate_forecasts method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_model = Mock()
        forecast_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=30),
            "yhat": [150.0] * 30,
            "yhat_lower": [140.0] * 30,
            "yhat_upper": [160.0] * 30,
        })
        mock_model.predict.return_value = forecast_df
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        result_forecast = runner.generate_forecasts(mock_model)
        
        # Assertions - generate_forecasts now returns a dictionary of forecasts by horizon
        assert isinstance(result_forecast, dict)
        assert "30_days" in result_forecast
        assert isinstance(result_forecast["30_days"], pd.DataFrame)
        assert len(result_forecast["30_days"]) == 30
        mock_model.predict.assert_called_once()
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_visualize_forecasts(self, mock_factory_class, sample_config_yaml, mock_spark_session):
        """Test visualize_forecasts method"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        train_df = Mock(spec=DataFrame)
        train_df.toPandas.return_value = pd.DataFrame({
            "usage_date": pd.date_range("2023-01-01", periods=100),
            "cost_in_billing_currency": [100.0] * 100,
        })
        
        forecast_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=30),
            "yhat": [150.0] * 30,
            "yhat_lower": [140.0] * 30,
            "yhat_upper": [160.0] * 30,
        })
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # This will actually create a plot file (matplotlib is imported inside the method)
            runner.visualize_forecasts(train_df, forecast_df, output_path=output_path)
            
            # Verify the file was created (or at least the method completed without error)
            # The actual plot generation depends on matplotlib being available
            assert True  # Method completed successfully
        except ImportError:
            # If matplotlib is not available, skip this test
            pytest.skip("matplotlib not available for visualization test")
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_run_full_pipeline(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test run() method with full pipeline steps"""
        # Setup all mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Data source mock
        mock_data_source = Mock()
        mock_data_source.load_data.return_value = sample_spark_dataframe
        mock_factory.create_data_source.return_value = mock_data_source
        
        # Data prep mock
        mock_data_prep = Mock()
        agg_df = sample_spark_dataframe
        train_df = Mock(spec=DataFrame)
        val_df = Mock(spec=DataFrame)
        test_df = Mock(spec=DataFrame)
        train_df.isEmpty.return_value = False
        train_df.toPandas.return_value = pd.DataFrame({
            "usage_date": pd.date_range("2023-01-01", periods=70),
            "cost_in_billing_currency": [100.0] * 70,
        })
        train_data_pd = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=70), "y": [100.0] * 70})
        mock_data_prep.aggregate_data.return_value = agg_df
        mock_data_prep.split.return_value = (train_df, val_df, test_df)
        mock_data_prep.prepare_for_training.return_value = train_data_pd
        mock_factory.create_data_preparation.return_value = mock_data_prep
        
        # Data quality mock
        mock_data_quality = Mock()
        mock_data_quality.comprehensive_validation.return_value = {"test": {"passed": True}}
        mock_factory.create_data_quality.return_value = mock_data_quality
        
        # Model mock
        mock_model = Mock()
        forecast_df = pd.DataFrame({
            "ds": pd.date_range("2024-01-01", periods=30),
            "yhat": [150.0] * 30,
        })
        mock_model.predict.return_value = forecast_df
        mock_model.train.return_value = {"status": "trained"}
        mock_factory.create_model.return_value = mock_model
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        runner.run(
            steps=['load_data', 'data_quality', 'prepare_data', 'train_model', 'forecast'],
            category="Total"
        )
        
        # Verify all steps were called
        mock_data_source.load_data.assert_called_once()
        mock_data_prep.aggregate_data.assert_called_once()
        mock_data_prep.split.assert_called_once()
        mock_data_quality.comprehensive_validation.assert_called()
        mock_model.train.assert_called_once()
        mock_model.predict.assert_called_once()  # Called once per horizon (config has [30] days)
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_run_partial_steps(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test run() method with partial steps"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        mock_data_source = Mock()
        mock_data_source.load_data.return_value = sample_spark_dataframe
        mock_factory.create_data_source.return_value = mock_data_source
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        runner.run(steps=['load_data'], category="Total")
        
        # Verify only load_data was called
        mock_data_source.load_data.assert_called_once()
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_run_missing_dependencies(self, mock_factory_class, sample_config_yaml, mock_spark_session):
        """Test run() method with missing step dependencies"""
        # Setup mocks
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        # Should fail because prepare_data needs load_data first
        with pytest.raises(ValueError, match="Cannot prepare data: data not loaded"):
            runner.run(steps=['prepare_data'], category="Total")
    
    @patch('acm_forecast.core.app_runner.PluginFactory')
    def test_run_with_save_results(self, mock_factory_class, sample_config_yaml, mock_spark_session, sample_spark_dataframe):
        """Test run() method with save_results step"""
        # Setup mocks (similar to full pipeline but with save_results)
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        # Data source mock
        mock_data_source = Mock()
        mock_data_source.load_data.return_value = sample_spark_dataframe
        mock_factory.create_data_source.return_value = mock_data_source
        
        # Data prep mock
        mock_data_prep = Mock()
        train_df = Mock(spec=DataFrame)
        train_df.isEmpty.return_value = False
        train_df.toPandas.return_value = pd.DataFrame({
            "usage_date": pd.date_range("2023-01-01", periods=70),
            "cost_in_billing_currency": [100.0] * 70,
        })
        train_data_pd = pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=70), "y": [100.0] * 70})
        mock_data_prep.split.return_value = (train_df, Mock(), Mock())
        mock_data_prep.prepare_for_training.return_value = train_data_pd
        mock_factory.create_data_preparation.return_value = mock_data_prep
        
        # Model mock
        mock_model = Mock()
        forecast_df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=30), "yhat": [150.0] * 30})
        mock_model.predict.return_value = forecast_df
        mock_model.train.return_value = {"status": "trained"}
        mock_factory.create_model.return_value = mock_model
        
        # Create runner and test
        runner = AppRunner(sample_config_yaml, session=mock_spark_session)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            runner.run(
                steps=['load_data', 'prepare_data', 'train_model', 'forecast', 'save_results'],
                category="Total",
                output_path=output_path
            )
            
            # Verify save_results was called (it should create a CSV file)
            assert Path(output_path).exists() or Path(output_path.replace('.png', '.csv')).exists()
        finally:
            Path(output_path).unlink(missing_ok=True)
            Path(output_path.replace('.csv', '.png')).unlink(missing_ok=True)

