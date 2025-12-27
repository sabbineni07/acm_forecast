"""
Integration tests for DataSource module with Delta tables
"""
import pytest
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.core import PluginFactory


@pytest.mark.integration
@pytest.mark.requires_spark
class TestDataSourceDelta:
    """Integration tests for DataSource with Delta table"""

    def test_load_from_delta_basic(self, spark_session, test_app_config):
        """Test basic data loading from Delta table"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")

        # Load all data
        df = data_source.load_data()

        assert df is not None
        assert not df.isEmpty(), "Should load some records"

        # Verify required columns exist
        columns = df.columns
        assert "usage_date" in columns
        assert "cost_in_billing_currency" in columns
        assert "quantity" in columns
        assert "meter_category" in columns
        assert "resource_location" in columns

    def test_load_from_delta_with_date_filter(self, spark_session, test_app_config):
        """Test loading data with date filters"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")

        # Load data for specific date range
        df = data_source.load_data(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )

        assert df is not None
        assert not df.isEmpty(), "Should load records in date range"

        # Verify dates are in range
        from datetime import date
        dates = df.select("usage_date").distinct().collect()
        for row in dates:
            date_val = row["usage_date"]
            # Convert to date if it's a datetime
            if hasattr(date_val, 'date'):
                date_val = date_val.date()
            # Compare with date objects
            assert date_val >= date(2024, 1, 1), f"Date {date_val} is before 2024-01-01"
            assert date_val <= date(2024, 1, 31), f"Date {date_val} is after 2024-01-31"

    def test_load_from_delta_with_category_filter(self, spark_session, test_app_config):
        """Test loading data with category filter"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")

        # First, find what categories exist in the test data
        all_df = data_source.load_data()
        available_categories = [row["meter_category"] for row in all_df.select("meter_category").distinct().collect()]
        assert len(available_categories) > 0, "Test data should have at least one category"

        # Use the first available category for the filter test
        test_category = available_categories[0]

        # Load data for specific category
        df = data_source.load_data(category=test_category)

        assert df is not None
        assert not df.isEmpty(), f"Should load records for {test_category} category"

        # Verify all records are the filtered category
        categories = df.select("meter_category").distinct().collect()
        category_values = [row["meter_category"] for row in categories]
        assert len(category_values) == 1, f"Expected 1 category, got {category_values}"
        assert category_values[0] == test_category, f"Expected '{test_category}', got '{category_values[0]}'"

    def test_get_data_profile(self, spark_session, test_app_config):
        """Test getting data profile"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")

        # Load sample data
        df = data_source.load_data()

        # Get profile
        profile = data_source.get_data_profile(df)

        assert profile is not None
        assert "total_records" in profile
        assert profile["total_records"] > 0
        assert "date_range" in profile
        assert "start" in profile["date_range"]
        assert "end" in profile["date_range"]
        assert "regional_distribution" in profile
        assert "category_distribution" in profile

    def test_validate_data_availability(self, spark_session, test_app_config):
        """Test data availability validation"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")

        # Load sample data
        df = data_source.load_data()

        # Validate availability
        validation = data_source.validate_data_availability(df)

        assert validation is not None
        assert "total_records" in validation
        assert validation["total_records"] > 0
        assert "data_freshness" in validation
        assert "meets_minimum_requirement" in validation
        assert validation["meets_minimum_requirement"] is True

    def test_map_attributes(self, test_app_config):
        """Test attribute mapping"""
        factory = PluginFactory()
        data_source = factory.create_data_source(test_app_config, spark=None, plugin_name="delta")

        mapping = data_source.map_attributes()

        assert mapping is not None
        assert "usage_date" in mapping
        assert "cost_in_billing_currency" in mapping
        assert "quantity" in mapping
        assert "meter_category" in mapping
        assert "resource_location" in mapping
        assert "plan_name" in mapping  # Replaced ServiceTier
        assert "effective_price" in mapping  # Replaced ResourceRate
