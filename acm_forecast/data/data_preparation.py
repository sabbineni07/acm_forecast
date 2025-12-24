"""
Data Preparation Module
Section 3.3: Data Preparation
"""

from typing import Optional, Tuple, List, Dict
import pandas as pd
from pyspark.sql import SparkSession, DataFrame, functions as sqlf
from pyspark.sql.functions import (
    col, sum as spark_sum, date_trunc, to_date, to_timestamp,
    when, isnan, isnull, lit, coalesce
)
from datetime import datetime, timedelta
import logging

from ..config import AppConfig

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Data preparation and preprocessing
    Section 3.3.1: Data Profile
    Section 3.3.2: Data Sampling
    Section 3.3.3: Data Treatment
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None):
        """
        Initialize data preparation
        
        Args:
            config: AppConfig instance containing configuration
            spark: SparkSession for Databricks environment
        """
        self.config = config
        self.spark = spark
        
    def aggregate_daily_costs(self, 
                            df: DataFrame,
                            group_by: Optional[List[str]] = None) -> DataFrame:
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
        
        # Base grouping columns (handle DATE type - convert to timestamp for date_trunc if needed)
        # usage_date is DATE type, so we can use it directly with to_date if it's string, or date_trunc if timestamp
        date_col_expr = col(self.config.feature.date_column)
        # If it's already a date type, use date_trunc; if string, convert first
        grouping_cols = [date_trunc("day", date_col_expr).alias("date")]
        grouping_cols.extend([col(col_name) for col_name in group_by])
        
        # Aggregate
        aggregated = df.groupBy(*grouping_cols).agg(
            spark_sum(self.config.feature.target_column).alias("daily_cost"),
            spark_sum("quantity").alias("total_quantity"),
            sqlf.avg("effective_price").alias("avg_rate")
        )
        
        # Rename date column back to usage_date for consistency
        aggregated = aggregated.withColumnRenamed("date", self.config.feature.date_column)
        
        logger.info(f"Aggregated to {aggregated.count()} daily records")
        return aggregated
    
    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values (Section 3.3.3)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values treated
        """
        # Forward-fill for missing days (assume zero cost if no usage)
        # This would require creating a complete date range and filling
        
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
            
        Returns:
            DataFrame with outlier flags
        """
        if method == "iqr":
            # IQR method
            # cost_stats = df.select(
            #     col(feature_config.target_column)
            # ).describe().toPandas()
            
            # Calculate IQR (would need to compute Q1, Q3)
            # For now, flag extreme values (> 99th percentile)
            percentile_99 = df.approxQuantile(
                self.config.feature.target_column, [0.99], 0.25
            )[0]
            
            df = df.withColumn(
                "is_outlier",
                col(self.config.feature.target_column) > percentile_99
            )
        
        logger.info("Outliers detected")
        return df
    
    def split_time_series(self, 
                         df: pd.DataFrame,
                         date_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data (Section 3.3.2)
        
        Args:
            df: Input DataFrame (must be sorted by date)
            date_col: Date column name
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        # Ensure sorted by date (convert to datetime if needed for sorting)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_split = self.config.training.train_split or 0.70
        validation_split = self.config.training.validation_split or 0.15
        train_end = int(n * train_split)
        val_end = train_end + int(n * validation_split)
        
        # Split chronologically
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(
            f"Split data: Train={len(train_df)}, "
            f"Validation={len(val_df)}, Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def prepare_for_prophet(self, 
                           df: pd.DataFrame,
                           date_col: str = None,
                           value_col: str = None) -> pd.DataFrame:
        """
        Prepare data for Prophet model (Section 4.2.1)
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Target value column
            
        Returns:
            DataFrame with 'ds' and 'y' columns for Prophet
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        if value_col is None:
            value_col = self.config.feature.target_column
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df[date_col]),
            "y": df[value_col]
        })
        
        # Remove any negative or zero values (Prophet requirement)
        prophet_df = prophet_df[prophet_df["y"] > 0]
        
        # Sort by date
        prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
        
        logger.info(f"Prepared {len(prophet_df)} records for Prophet")
        return prophet_df
    
    def prepare_for_arima(self,
                         df: pd.DataFrame,
                         date_col: str = None,
                         value_col: str = None) -> pd.Series:
        """
        Prepare data for ARIMA model (Section 4.2.1)
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Target value column
            
        Returns:
            Time series as pandas Series with datetime index
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        if value_col is None:
            value_col = self.config.feature.target_column
        # Create time series (ensure date_col is datetime for indexing)
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        ts = df_copy.set_index(date_col)[value_col].sort_index()
        
        # Remove any NaN values
        ts = ts.dropna()
        
        logger.info(f"Prepared {len(ts)} records for ARIMA")
        return ts
    
    def segment_by_category(self, 
                           df: DataFrame) -> Dict[str, DataFrame]:
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
            if category_df.count() > 0:
                segments[category] = category_df
        
        logger.info(f"Segmented into {len(segments)} categories")
        return segments


