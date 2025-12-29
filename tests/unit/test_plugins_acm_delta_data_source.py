"""
Unit tests for ACMDeltaDataSource plugin
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

from acm_forecast.core.plugin_registry import PluginFactory
from acm_forecast.config import AppConfig


class TestACMDeltaDataSourcePlugin:
    """Unit tests for ACMDeltaDataSource plugin"""
    
    @pytest.mark.unit
    def test_acm_delta_data_source_initialization(self, sample_app_config, spark_session):
        """Test ACMDeltaDataSource initialization"""
        factory = PluginFactory()
        data_source = factory.create_data_source(
            sample_app_config,
            spark=spark_session,
            plugin_name="acm"
        )
        
        assert data_source is not None
        assert data_source.config == sample_app_config
        assert data_source.spark == spark_session
    
    @pytest.mark.unit
    def test_acm_delta_data_source_has_required_methods(self, sample_app_config, spark_session):
        """Test that ACMDeltaDataSource has all required interface methods"""
        factory = PluginFactory()
        data_source = factory.create_data_source(
            sample_app_config,
            spark=spark_session,
            plugin_name="acm"
        )
        
        # Check IDataSource interface methods
        assert hasattr(data_source, 'load_data')
        assert hasattr(data_source, 'get_data_profile')
        assert hasattr(data_source, 'validate_data_availability')
        assert hasattr(data_source, 'map_attributes')
    
    @pytest.mark.unit
    def test_generate_sample_data(self, sample_app_config, spark_session):
        """Test generating sample data"""
        factory = PluginFactory()
        data_source = factory.create_data_source(
            sample_app_config,
            spark=spark_session,
            plugin_name="acm"
        )
        
        # Configure for sample data generation
        sample_app_config.data.generate_sample_data = True
        sample_app_config.data.sample_data_days = 30
        sample_app_config.data.sample_data_records_per_day = 10
        sample_app_config.data.sample_data_subscriptions = 2
        sample_app_config.data.sample_data_start_date = "2024-01-01"
        
        df = data_source.generate_sample_data()
        
        assert df is not None
        assert not df.isEmpty()
        
        # Check required columns
        columns = df.columns
        assert sample_app_config.feature.date_column in columns
        assert sample_app_config.feature.target_column in columns

