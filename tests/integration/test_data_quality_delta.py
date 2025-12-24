"""
Integration tests for DataQualityValidator module with Delta tables
"""
import pytest
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.data.data_source import DataSource
from acm_forecast.data.data_quality import DataQualityValidator


@pytest.mark.integration
@pytest.mark.requires_spark
class TestDataQualityDelta:
    """Integration tests for DataQualityValidator with Delta table"""
    
    def test_validate_completeness(self, spark_session, test_app_config):
        """Test completeness validation"""
        data_source = DataSource(test_app_config, spark_session)
        quality_validator = DataQualityValidator(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Validate completeness
        results = quality_validator.validate_completeness(df)

        assert results is not None
        assert "total_records" in results
        assert results["total_records"] > 0
        assert "missing_values" in results
        assert "completeness_rate" in results
        assert "meets_threshold" in results

        # Completeness rate should be reasonable for test data
        assert 0 <= results["completeness_rate"] <= 100
    
    def test_validate_accuracy(self, spark_session, test_app_config):
        """Test accuracy validation"""
        data_source = DataSource(test_app_config, spark_session)

        quality_validator = DataQualityValidator(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Validate accuracy
        results = quality_validator.validate_accuracy(df)
        
        assert results is not None
        assert "negative_costs" in results
        assert "zero_costs" in results
        assert "date_range" in results
        assert "data_quality_issues" in results
        
        # Test data should not have negative costs
        assert results["negative_costs"] == 0
    
    def test_validate_consistency(self, spark_session, test_app_config):
        """Test consistency validation"""
        data_source = DataSource(test_app_config, spark_session)
        quality_validator = DataQualityValidator(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Validate consistency
        results = quality_validator.validate_consistency(df)

        assert results is not None
        assert "total_records" in results
        assert "distinct_records" in results
        assert "duplicate_records" in results
        assert "duplicate_percentage" in results

        # Duplicate percentage should be reasonable
        assert 0 <= results["duplicate_percentage"] <= 100

    def test_validate_timeliness(self, spark_session, test_app_config):
        """Test timeliness validation"""
        data_source = DataSource(test_app_config, spark_session)
        quality_validator = DataQualityValidator(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Validate timeliness
        results = quality_validator.validate_timeliness(df)

        assert results is not None
        assert "latest_date" in results
        assert "hours_since_update" in results or results["latest_date"] is None
        assert "is_fresh" in results
        assert "meets_sla" in results

    def test_comprehensive_validation(self, spark_session, test_app_config):
        """Test comprehensive data quality validation"""
        data_source = DataSource(test_app_config, spark_session)
        quality_validator = DataQualityValidator(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Run comprehensive validation
        results = quality_validator.comprehensive_validation(df)

        assert results is not None
        assert "completeness" in results
        assert "accuracy" in results
        assert "consistency" in results
        assert "timeliness" in results
        assert "quality_score" in results

        # Quality score should be between 0 and 100
        assert 0 <= results["quality_score"] <= 100

