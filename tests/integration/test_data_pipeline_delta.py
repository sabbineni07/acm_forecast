"""
Integration tests for complete data pipeline with Delta tables
Tests data_source → data_quality → data_preparation → feature_engineering flow
"""
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from acm_forecast.config import AppConfig
from acm_forecast.data.data_source import DataSource
from acm_forecast.data.data_quality import DataQualityValidator
from acm_forecast.data.data_preparation import DataPreparation
from acm_forecast.data.feature_engineering import FeatureEngineer


@pytest.mark.integration
@pytest.mark.requires_spark
@pytest.mark.filterwarnings("ignore:unclosed <socket.socket")
class TestDataPipelineDelta:
    """Integration tests for complete data pipeline with Delta table"""

    def test_end_to_end_data_processing(self, spark_session, test_app_config):
        """Test complete data processing pipeline"""
        # Initialize components
        data_source = DataSource(test_app_config, spark_session)
        data_quality = DataQualityValidator(test_app_config, spark_session)
        data_prep = DataPreparation(test_app_config, spark_session)
        feature_engineer = FeatureEngineer(test_app_config, spark_session)

        # Step 1: Load data from Delta
        print("Step 1: Loading data from Delta table...")
        df_spark = data_source.load_from_delta()
        assert df_spark.count() > 0

        # Step 2: Validate data quality
        print("Step 2: Validating data quality...")
        quality_results = data_quality.comprehensive_validation(df_spark)
        assert quality_results["quality_score"] >= 0

        # Step 3: Aggregate daily costs
        print("Step 3: Aggregating daily costs...")
        daily_df_spark = data_prep.aggregate_daily_costs(df_spark, group_by=["meter_category"])
        daily_count = daily_df_spark.count()
        assert daily_count > 0

        # Step 4: Convert to Pandas for feature engineering
        print("Step 4: Converting to Pandas...")
        daily_df = daily_df_spark.toPandas()
        assert len(daily_df) > 0
        
        # Convert date column from object/date to datetime64[ns] for pandas
        date_col = test_app_config.feature.date_column
        if date_col in daily_df.columns:
            # Convert date column to datetime, handling both date objects and strings
            daily_df[date_col] = pd.to_datetime(daily_df[date_col], errors='coerce').astype('datetime64[ns]')

        # Step 5: Split data
        print("Step 5: Splitting data...")
        train_df, val_df, test_df = data_prep.split_time_series(daily_df)
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Step 6: Feature engineering
        print("Step 6: Feature engineering...")
        features_df = feature_engineer.prepare_xgboost_features(train_df)
        assert len(features_df) > 0
        assert len(features_df.columns) > len(train_df.columns)

        print("✅ End-to-end data processing pipeline completed successfully")

    def test_aggregate_daily_costs_with_delta(self, spark_session, test_app_config):
        """Test daily aggregation with Delta table data"""
        data_source = DataSource(test_app_config, spark_session)
        data_prep = DataPreparation(test_app_config, spark_session)

        # Load data
        df = data_source.load_from_delta()

        # Aggregate by category
        daily_df = data_prep.aggregate_daily_costs(df, group_by=["meter_category"])

        assert daily_df.count() > 0
        
        # Verify aggregation columns
        columns = daily_df.columns
        assert test_app_config.feature.date_column in columns
        assert test_app_config.feature.target_column in columns  # Renamed from daily_cost
        assert "total_quantity" in columns
        assert "avg_rate" in columns
        assert "meter_category" in columns

    def test_data_preparation_with_delta_data(self, spark_session, test_app_config):
        """Test data preparation with Delta table data"""
        import pandas as pd

        data_source = DataSource(test_app_config, spark_session)
        data_prep = DataPreparation(test_app_config, spark_session)

        # Load and aggregate
        df = data_source.load_from_delta()
        daily_df_spark = data_prep.aggregate_daily_costs(df)
        daily_df = daily_df_spark.toPandas()

        # Convert date column to datetime64[ns] for pandas
        date_col = test_app_config.feature.date_column
        if date_col in daily_df.columns:
            daily_df[date_col] = pd.to_datetime(daily_df[date_col], errors='coerce').astype('datetime64[ns]')

        # Test Prophet preparation
        prophet_df = data_prep.prepare_for_prophet(daily_df)
        assert len(prophet_df) > 0
        assert "ds" in prophet_df.columns
        assert "y" in prophet_df.columns

        # Test ARIMA preparation
        arima_ts = data_prep.prepare_for_arima(daily_df)
        assert len(arima_ts) > 0

