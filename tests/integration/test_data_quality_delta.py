"""
Integration tests for DataQualityValidator module with Delta tables
"""
import pytest
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.core import PluginFactory


@pytest.mark.integration
@pytest.mark.requires_spark
class TestDataQualityDelta:
    """Integration tests for DataQualityValidator with Delta table"""

    def test_validate_completeness(self, spark_session, test_app_config):
        """Test completeness validation"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        
        # Create test data with some missing values
        factory_ds = PluginFactory()
        data_source = factory_ds.create_data_source(test_app_config, spark_session, plugin_name="acm")
        df = data_source.load_data()

        # Validate completeness
        results = data_quality.validate_completeness(df)

        assert results is not None
        assert "total_records" in results
        assert "missing_values" in results
        assert "completeness_rate" in results
        assert "meets_threshold" in results
        assert results["total_records"] > 0

    def test_validate_accuracy(self, spark_session, test_app_config):
        """Test accuracy validation"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        
        factory_ds = PluginFactory()
        data_source = factory_ds.create_data_source(test_app_config, spark_session, plugin_name="acm")
        df = data_source.load_data()

        # Validate accuracy
        results = data_quality.validate_accuracy(df)

        assert results is not None
        assert "negative_costs" in results
        assert "zero_costs" in results
        assert "non_usd_currency" in results
        assert "date_range" in results
        assert "data_quality_issues" in results

    def test_validate_consistency(self, spark_session, test_app_config):
        """Test consistency validation"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        
        factory_ds = PluginFactory()
        data_source = factory_ds.create_data_source(test_app_config, spark_session, plugin_name="acm")
        df = data_source.load_data()

        # Validate consistency
        results = data_quality.validate_consistency(df)

        assert results is not None
        assert "total_records" in results
        assert "distinct_records" in results
        assert "duplicate_records" in results
        assert "duplicate_percentage" in results

    def test_validate_timeliness(self, spark_session, test_app_config):
        """Test timeliness validation"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        
        factory_ds = PluginFactory()
        data_source = factory_ds.create_data_source(test_app_config, spark_session, plugin_name="acm")
        df = data_source.load_data()

        # Validate timeliness
        results = data_quality.validate_timeliness(df)

        assert results is not None
        assert "latest_date" in results
        assert "hours_since_update" in results
        assert "is_fresh" in results
        assert "meets_sla" in results

    def test_comprehensive_validation(self, spark_session, test_app_config):
        """Test comprehensive validation"""
        factory = PluginFactory()
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        
        factory_ds = PluginFactory()
        data_source = factory_ds.create_data_source(test_app_config, spark_session, plugin_name="acm")
        df = data_source.load_data()

        # Run comprehensive validation
        results = data_quality.comprehensive_validation(df)

        assert results is not None
        assert "completeness" in results
        assert "accuracy" in results
        assert "consistency" in results
        assert "timeliness" in results
        assert "quality_score" in results
        assert 0 <= results["quality_score"] <= 100
