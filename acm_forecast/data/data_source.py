"""
Data Source Module
Section 3.1: Data Sourcing
"""

from typing import Optional, Dict, Any
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, sum as spark_sum, avg, count, min as spark_min, max as spark_max
import logging
from datetime import date

from ..config import AppConfig

logger = logging.getLogger(__name__)


class DataSource:
    """
    Data source handler for Azure Cost Management data
    Section 3.1.1: Data Source Location
    Section 3.1.2: Data Constraints
    Section 3.1.3: Data Mapping
    Section 3.1.4: Data Reliability
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None):
        """
        Initialize data source
        
        Args:
            config: AppConfig instance containing configuration
            spark: SparkSession for Databricks environment
        """
        self.config = config
        self.spark = spark
        self.delta_table_path = config.data.delta_table_path
        self.database_name = config.data.database_name
        self.table_name = config.data.table_name
        
    def load_from_delta(self, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       category: Optional[str] = None) -> DataFrame:
        """
        Load data from Delta table (Section 3.1.1)
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            category: MeterCategory filter
            
        Returns:
            Spark DataFrame with Azure cost data
        """
        if self.spark is None:
            raise ValueError("SparkSession is required for Delta table access")
        
        logger.info(f"Loading data from {self.delta_table_path}")
        
        # Read from Delta table
        df = self.spark.read.format("delta").table(self.delta_table_path)
        
        # Apply filters (using snake_case column names)
        date_col = self.config.feature.date_column
        if start_date:
            df = df.filter(col(date_col) >= start_date)
        if end_date:
            df = df.filter(col(date_col) <= end_date)
        if category:
            df = df.filter(col("meter_category") == category)
        
        logger.info(f"Loaded {df.count()} records")
        return df
    
    def get_data_profile(self, df: DataFrame) -> Dict[str, Any]:
        """
        Get data profile (Section 3.3.1)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data profile information
        """
        total_records = df.count()
        
        # Date range (using snake_case)
        date_col = self.config.feature.date_column
        target_col = self.config.feature.target_column
        date_stats = df.select(
            spark_min(col(date_col)).alias("min_date"),
            spark_max(col(date_col)).alias("max_date")
        ).collect()[0].asDict()
        
        # Regional distribution
        region_dist = df.groupBy("resource_location").agg(
            count("*").alias("count"),
            spark_sum(target_col).alias("total_cost")
        ).toPandas()
        
        # Category distribution
        category_dist = df.groupBy("meter_category").agg(
            count("*").alias("count"),
            spark_sum(target_col).alias("total_cost")
        ).toPandas()
        
        profile = {
            "total_records": total_records,
            "date_range": {
                "start": date_stats.get("min_date"),
                "end": date_stats.get("max_date")
            },
            "regional_distribution": region_dist.to_dict("records"),
            "category_distribution": category_dist.to_dict("records")
        }
        
        return profile
    
    def validate_data_availability(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data availability and constraints (Section 3.1.2)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation results
        """
        total_records = df.count()
        
        # Check for missing dates
        date_col = self.config.feature.date_column
        date_range = df.agg(
            spark_min(col(date_col)).alias("min_date"),
            spark_max(col(date_col)).alias("max_date")
        ).collect()[0].asDict()
        
        # Check data freshness (Section 3.1.4)
        latest_date = date_range.get("max_date")
        if latest_date and self.config.data.max_data_delay_hours:
            hours_since_update = (date.today() - latest_date).total_seconds() / 3600
            is_fresh = hours_since_update <= self.config.data.max_data_delay_hours
        else:
            is_fresh = False
            hours_since_update = None
        
        validation = {
            "total_records": total_records,
            "data_freshness": {
                "latest_date": latest_date,
                "hours_since_update": hours_since_update,
                "is_fresh": is_fresh
            },
            "meets_minimum_requirement": total_records > 0
        }
        
        return validation
    
    def map_attributes(self) -> Dict[str, str]:
        """
        Get attribute mapping (Section 3.1.3)
        Returns snake_case column names matching Delta table schema
        
        Returns:
            Dictionary mapping framework names to Delta table column names (snake_case)
        """
        return {
            "subscription_id": "subscription_id",
            "resource_group": "resource_group",
            "resource_location": "resource_location",
            "usage_date": "usage_date",  # DATE type
            "meter_category": "meter_category",
            "meter_sub_category": "meter_sub_category",
            "meter_id": "meter_id",
            "meter_name": "meter_name",
            "meter_region": "meter_region",
            "quantity": "quantity",
            "effective_price": "effective_price",
            "cost_in_billing_currency": "cost_in_billing_currency",  # Target variable
            "consumed_service": "consumed_service",
            "resource_id": "resource_id",
            "tags": "tags",
            "offer_id": "offer_id",
            "additional_info": "additional_info",
            "service_info1": "service_info1",
            "service_info2": "service_info2",
            "product_name": "product_name",
            "plan_name": "plan_name",  # Replaces ServiceTier
            "billing_currency_code": "billing_currency_code",
            "unit_of_measure": "unit_of_measure"
        }


