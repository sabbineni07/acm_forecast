"""
Unit tests for DefaultDataQuality plugin
"""
import pytest
from unittest.mock import Mock, MagicMock
from pyspark.sql import SparkSession

from acm_forecast.core.plugin_registry import PluginFactory
from acm_forecast.config import AppConfig


class TestDefaultDataQualityPlugin:
    """Unit tests for DefaultDataQuality plugin"""
    
    @pytest.mark.unit
    def test_default_data_quality_initialization(self, sample_app_config, spark_session):
        """Test DefaultDataQuality initialization"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(
            sample_app_config,
            spark=spark_session,
            plugin_name="default"
        )
        
        assert data_quality is not None
        assert data_quality.config == sample_app_config
        assert data_quality.spark == spark_session
    
    @pytest.mark.unit
    def test_default_data_quality_has_required_methods(self, sample_app_config, spark_session):
        """Test that DefaultDataQuality has all required interface methods"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(
            sample_app_config,
            spark=spark_session,
            plugin_name="default"
        )
        
        # Check IDataQuality interface methods
        assert hasattr(data_quality, 'validate_completeness')
        assert hasattr(data_quality, 'validate_accuracy')
        assert hasattr(data_quality, 'validate_consistency')
        assert hasattr(data_quality, 'validate_timeliness')
        assert hasattr(data_quality, 'comprehensive_validation')

