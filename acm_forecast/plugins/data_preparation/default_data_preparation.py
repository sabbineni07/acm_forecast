"""
Default Data Preparation Plugin

PRIMARY IMPLEMENTATION for data preparation and preprocessing.
The actual implementation is here - DataPreparation class delegates to this.
"""

from typing import Dict, Any, Optional, List, Tuple
from pyspark.sql import SparkSession, DataFrame, functions as sqlf
from pyspark.sql.functions import (
    col, sum as spark_sum, date_trunc, to_date, to_timestamp,
    when, isnan, isnull, lit, coalesce, row_number, count
)
from pyspark.sql.window import Window
import pandas as pd
from datetime import datetime, timedelta
import logging

from ...core.interfaces import IDataPreparation
from ...core.base_plugin import BasePlugin
from ...config import AppConfig

logger = logging.getLogger(__name__)


class DefaultDataPreparation(BasePlugin, IDataPreparation):
    """
    Default data preparation plugin - PRIMARY IMPLEMENTATION
    Section 3.3.1: Data Profile
    Section 3.3.2: Data Sampling
    Section 3.3.3: Data Treatment
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize default data preparation plugin"""
        super().__init__(config, spark, **kwargs)
    
    def aggregate_data(self, df: DataFrame, group_by: Optional[List[str]] = None) -> DataFrame:
        """
        Aggregate costs to daily level (Section 3.3.3)
        
        Args:
            df: Input DataFrame
            group_by: Additional grouping columns (e.g., ['meter_category'])
            
        Returns:
            Aggregated DataFrame with daily costs
        """
        if group_by is None:
            group_by = []
        
        # Base grouping columns (handle DATE type - convert to timestamp for date_trunc, then back to date)
        # usage_date is DATE type, so we use date_trunc which returns timestamp, then convert back to date
        date_col_expr = col(self.config.feature.date_column)
        # date_trunc returns timestamp, convert back to date for proper Pandas handling
        grouping_cols = [to_date(date_trunc("day", date_col_expr)).alias("date")]
        grouping_cols.extend([col(col_name) for col_name in group_by])
        
        # Aggregate
        target_col = self.config.feature.target_column
        aggregated = df.groupBy(*grouping_cols).agg(
            spark_sum(target_col).alias("daily_cost"),
            spark_sum("quantity").alias("total_quantity"),
            sqlf.avg("effective_price").alias("avg_rate")
        )
        
        # Rename date column back to usage_date for consistency
        aggregated = aggregated.withColumnRenamed("date", self.config.feature.date_column)
        # Rename daily_cost back to target column name for downstream processing
        aggregated = aggregated.withColumnRenamed("daily_cost", target_col)
        
        logger.info("Aggregated to daily records")
        return aggregated
    
    def prepare_for_training(self, df: DataFrame, model_type: str = "prophet") -> pd.DataFrame:
        """
        Prepare data for model training
        
        Args:
            df: Input Spark DataFrame
            model_type: Type of model (prophet, arima, xgboost)
            
        Returns:
            Prepared pandas DataFrame (converted from Spark DataFrame at the end)
        """
        # Keep Spark DataFrame and do operations in Spark, convert to pandas at the end
        if model_type == "prophet":
            return self._prepare_for_prophet(df)
        elif model_type == "arima":
            return self._prepare_for_arima(df)
        elif model_type == "xgboost":
            return self._prepare_for_xgboost(df)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _prepare_for_prophet(self, 
                           df: DataFrame,
                           date_col: str = None,
                           value_col: str = None) -> pd.DataFrame:
        """
        Prepare data for Prophet model (Section 4.2.1)
        
        Args:
            df: Input Spark DataFrame
            date_col: Date column name
            value_col: Target value column
            
        Returns:
            Pandas DataFrame with 'ds' and 'y' columns for Prophet
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        if value_col is None:
            value_col = self.config.feature.target_column
        
        # Select and rename columns in Spark (distributed operation)
        prophet_df_spark = df.select(
            col(date_col).alias("ds"),
            col(value_col).alias("y")
        )
        
        # Filter out negative or zero values in Spark (Prophet requirement)
        prophet_df_spark = prophet_df_spark.filter(col("y") > 0)
        
        # Sort by date in Spark
        prophet_df_spark = prophet_df_spark.orderBy("ds")
        
        # Convert to pandas only at the end (after all Spark operations)
        prophet_df = prophet_df_spark.toPandas()
        
        # Ensure datetime type for 'ds' column (final step in pandas)
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        
        logger.info(f"Prepared {len(prophet_df)} records for Prophet")
        return prophet_df
    
    def _prepare_for_arima(self,
                         df: DataFrame,
                         date_col: str = None,
                         value_col: str = None) -> pd.Series:
        """
        Prepare data for ARIMA model (Section 4.2.1)
        
        Args:
            df: Input Spark DataFrame
            date_col: Date column name
            value_col: Target value column
            
        Returns:
            Time series as pandas Series with datetime index
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        if value_col is None:
            value_col = self.config.feature.target_column
        
        # Select columns and filter out NaN/null values in Spark (distributed operation)
        arima_df_spark = df.select(
            col(date_col).alias("date"),
            col(value_col).alias("value")
        ).filter(
            col("value").isNotNull()
        )
        
        # Sort by date in Spark
        arima_df_spark = arima_df_spark.orderBy("date")
        
        # Convert to pandas only at the end (after all Spark operations)
        arima_df = arima_df_spark.toPandas()
        
        # Create time series with datetime index (final step in pandas)
        arima_df["date"] = pd.to_datetime(arima_df["date"])
        ts = arima_df.set_index("date")["value"].sort_index()
        
        logger.info(f"Prepared {len(ts)} records for ARIMA")
        return ts
    
    def _prepare_for_xgboost(self, df: DataFrame) -> pd.DataFrame:
        """
        Prepare data for XGBoost - returns DataFrame directly (feature engineering handled separately)
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            Pandas DataFrame (converted from Spark DataFrame)
        """
        # For XGBoost, feature engineering is handled separately by IFeatureEngineer
        # Just convert to pandas here (minimal processing needed)
        return df.toPandas()
    
    # Additional methods for backward compatibility (not in interface but used by original class)
    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values (Section 3.3.3)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values treated
        """
        # Handle missing numerical values
        numerical_cols = [self.config.feature.target_column, "quantity", "effective_price"]
        for col_name in numerical_cols:
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    coalesce(col(col_name), lit(0.0))
                )
        
        # Handle missing categorical values
        categorical_cols = ["meter_category", "resource_location", "plan_name"]
        for col_name in categorical_cols:
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    when(isnull(col(col_name)) | isnan(col(col_name)), "Unknown")
                    .otherwise(col(col_name))
                )
        
        logger.info("Missing values handled")
        return df
    
    def detect_outliers(self, df: DataFrame, 
                      method: str = "iqr") -> DataFrame:
        """
        Detect outliers (Section 3.3.3)
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr' or 'zscore')

            Interquartile Range (IQR) identifies outliers by establishing "fences" around the middle 50% of data
        Returns:
            DataFrame with outlier flags
        """
        if method == "iqr":
            # Calculate IQR - flag extreme values (> 99th percentile)
            percentile_99 = df.approxQuantile(
                self.config.feature.target_column, [0.99], 0.25
            )[0]
            
            df = df.withColumn(
                "is_outlier",
                col(self.config.feature.target_column) > percentile_99
            )
        
        logger.info("Outliers detected")
        return df
    
    def split(self, df: DataFrame, date_col: Optional[str] = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Split data into train, validation, and test sets (Section 3.3.2)
        
        Args:
            df: Input Spark DataFrame (will be sorted by date)
            date_col: Date column name
            
        Returns:
            Tuple of (train, validation, test) Spark DataFrames
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        
        spark = self.validate_spark()
        
        # Sort by date and add row number
        window = Window.orderBy(col(date_col))
        df_with_row = df.withColumn("row_num", row_number().over(window))
        
        # Get total count
        total_count = df_with_row.count()
        
        # Calculate split indices
        train_split = self.config.training.train_split or 0.70
        validation_split = self.config.training.validation_split or 0.15
        train_end = int(total_count * train_split)
        val_end = train_end + int(total_count * validation_split)
        
        # Split chronologically using row numbers
        train_df = df_with_row.filter(col("row_num") <= train_end).drop("row_num")
        val_df = df_with_row.filter((col("row_num") > train_end) & (col("row_num") <= val_end)).drop("row_num")
        test_df = df_with_row.filter(col("row_num") > val_end).drop("row_num")
        
        # Log split info (avoid expensive count() calls)
        logger.info(
            f"Split data: Train (0-{train_end}), "
            f"Validation ({train_end+1}-{val_end}), Test ({val_end+1}-{total_count})"
        )
        
        return train_df, val_df, test_df
    
    def segment_by_category(self, df: DataFrame) -> Dict[str, DataFrame]:
        """
        Segment data by category (Section 3.3.5)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of DataFrames by category
        """
        segments = {}
        
        categories = self.config.data.cost_categories or []
        for category in categories:
            category_df = df.filter(col("meter_category") == category)
            if not category_df.isEmpty():
                segments[category] = category_df
        
        logger.info(f"Segmented into {len(segments)} categories")
        return segments
