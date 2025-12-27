"""
Integration tests for complete data pipeline with Delta tables
Tests data_source → data_quality → data_preparation → feature_engineering flow
"""
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.core import PluginFactory


@pytest.mark.integration
@pytest.mark.requires_spark
class TestDataPipelineDelta:
    """Integration tests for complete data pipeline with Delta table"""

    def test_end_to_end_data_processing(self, spark_session, test_app_config):
        """Test complete data processing pipeline"""
        factory = PluginFactory()
        
        # Step 1: Load data
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")
        df_spark = data_source.load_data()

        # Step 2: Data quality validation
        data_quality = factory.create_data_quality(test_app_config, spark_session, plugin_name="default")
        quality_results = data_quality.comprehensive_validation(df_spark)
        assert quality_results["quality_score"] >= 0

        # Step 3: Data preparation - aggregate daily costs
        data_prep = factory.create_data_preparation(test_app_config, spark_session, plugin_name="default")
        daily_df_spark = data_prep.aggregate_data(df_spark)

        # Step 4: Converting to Pandas...
        daily_df = daily_df_spark.toPandas()
        # Explicitly convert date column to datetime64[ns] to avoid dtype issues
        daily_df[test_app_config.feature.date_column] = pd.to_datetime(
            daily_df[test_app_config.feature.date_column]
        ).astype('datetime64[ns]')

        # Step 5: Split data
        train_df, val_df, test_df = data_prep.split(daily_df)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(val_df) + len(test_df) == len(daily_df)

        # Step 6: Feature engineering
        feature_engineer = factory.create_feature_engineer(test_app_config, spark_session, plugin_name="default")
        train_features = feature_engineer.create_temporal_features(train_df)
        assert len(train_features.columns) > len(train_df.columns)

    def test_aggregate_data_with_delta(self, spark_session, test_app_config):
        """Test daily aggregation with Delta table data"""
        factory = PluginFactory()
        
        # Load data
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")
        df = data_source.load_data()

        # Aggregate
        data_prep = factory.create_data_preparation(test_app_config, spark_session, plugin_name="default")
        daily_df = data_prep.aggregate_data(df)

        assert daily_df is not None
        assert not daily_df.isEmpty()

        # Verify aggregated columns exist
        columns = daily_df.columns
        assert test_app_config.feature.date_column in columns
        assert test_app_config.feature.target_column in columns

    def test_data_preparation_with_delta_data(self, spark_session, test_app_config):
        """Test data preparation with Delta table data"""
        factory = PluginFactory()
        
        # Load and aggregate data
        data_source = factory.create_data_source(test_app_config, spark_session, plugin_name="delta")
        df = data_source.load_data()
        
        data_prep = factory.create_data_preparation(test_app_config, spark_session, plugin_name="default")
        daily_df_spark = data_prep.aggregate_data(df)

        # Convert to Pandas
        daily_df = daily_df_spark.toPandas()
        # Explicitly convert date column to datetime64[ns]
        daily_df[test_app_config.feature.date_column] = pd.to_datetime(
            daily_df[test_app_config.feature.date_column]
        ).astype('datetime64[ns]')

        # Prepare for Prophet
        prophet_data = data_prep.prepare_for_training(daily_df, model_type="prophet")
        assert "ds" in prophet_data.columns
        assert "y" in prophet_data.columns

        # Prepare for ARIMA
        arima_data = data_prep.prepare_for_training(daily_df, model_type="arima")
        assert len(arima_data) > 0
        assert hasattr(arima_data, 'index')  # Should be a Series with datetime index
