"""
Delta Data Source Plugin

Primary implementation for loading data from Delta tables.
This is the source of truth - original DataSource class delegates to this.
"""

from typing import Dict, Any, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, sum as spark_sum, avg, count, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
import logging
from datetime import date, datetime, timedelta
import uuid
import random

from ...core.interfaces import IDataSource
from ...core.base_plugin import BasePlugin
from ...config import AppConfig

logger = logging.getLogger(__name__)

# Azure regions for sample data generation
AZURE_REGIONS = ["East US", "East US 2", "South Central US"]

# Meter categories (matching framework expectations)
METER_CATEGORIES = {
    "Compute": ["Virtual Machines", "Container Instances", "App Service", "Functions"],
    "Storage": ["Blob Storage", "File Storage", "Table Storage", "Queue Storage", "Disk Storage"],
    "Network": ["Bandwidth", "Load Balancer", "VPN Gateway", "ExpressRoute", "DNS"],
    "Database": ["SQL Database", "Cosmos DB", "PostgreSQL", "MySQL", "Redis Cache"],
    "Analytics": ["Databricks", "HDInsight", "Synapse Analytics", "Stream Analytics"],
    "AI/ML": ["Cognitive Services", "Azure ML", "Bot Services", "Computer Vision"],
    "Security": ["Key Vault", "Active Directory", "Security Center", "DDoS Protection"],
    "Management": ["Monitor", "Log Analytics", "Automation", "Backup", "Site Recovery"]
}

# Service tiers (for plan_name)
SERVICE_TIERS = ["Basic", "Standard", "Premium", "Free", "Enterprise"]

# Currency
CURRENCY = "USD"

# Units by category
UNITS = {
    "Compute": ["Hours", "vCPU-hours"],
    "Storage": ["GB-Month", "GB", "TB"],
    "Network": ["GB", "MB"],
    "Database": ["DTU-hours", "vCore-hours", "GB-Month"],
    "Analytics": ["DBU-hours", "Node-hours"],
    "AI/ML": ["Transactions", "Hours"],
    "Security": ["Hours", "Transactions"],
    "Management": ["GB", "Hours"]
}


class DeltaDataSource(BasePlugin, IDataSource):
    """
    Delta table data source plugin - PRIMARY IMPLEMENTATION
    Section 3.1.1: Data Source Location
    Section 3.1.2: Data Constraints
    Section 3.1.3: Data Mapping
    Section 3.1.4: Data Reliability
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize Delta data source plugin
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Plugin-specific configuration
        """
        super().__init__(config, spark, **kwargs)
        self.delta_table_path = config.data.delta_table_path
        self.database_name = config.data.database_name
        self.table_name = config.data.table_name
    
    @staticmethod
    def _generate_subscription_id() -> str:
        """Generate a synthetic Azure subscription ID (GUID format)"""
        return str(uuid.uuid4())
    
    @staticmethod
    def _generate_date_range(start_date: datetime, days: int):
        """Generate a list of dates"""
        return [start_date + timedelta(days=i) for i in range(days)]
    
    @staticmethod
    def _generate_time_series_cost(base_cost: float, trend: float, seasonality: bool = True) -> float:
        """
        Generate realistic cost with trend and seasonality
        
        Args:
            base_cost: Base cost value
            trend: Trend coefficient (e.g., 0.001 for 0.1% growth per day)
            seasonality: Add weekly seasonality
            
        Returns:
            Generated cost value (as Python float)
        """
        # Add trend
        cost = base_cost * (1 + trend)
        
        # Add weekly seasonality (lower costs on weekends)
        if seasonality:
            day_of_week = random.randint(0, 6)
            if day_of_week >= 5:  # Weekend
                cost *= 0.7
            else:  # Weekday
                cost *= 1.1
        
        # Add random variation (using Python random to avoid numpy types)
        cost *= random.uniform(0.8, 1.2)
        
        # Ensure positive and convert to Python float
        return float(max(cost, 0.01))
    
    def generate_sample_data(self) -> DataFrame:
        """
        Generate synthetic Azure cost management data with REQUIRED columns only
        Uses configuration from self.config.data for generation parameters
        
        Returns:
            PySpark DataFrame with synthetic cost data using snake_case column names.
        """
        spark = self.validate_spark()
        
        # Get generation parameters from config
        days = self.config.data.sample_data_days or 365
        records_per_day = self.config.data.sample_data_records_per_day or 100
        subscription_count = self.config.data.sample_data_subscriptions or 3
        
        # Parse start date
        if self.config.data.sample_data_start_date:
            start_date = datetime.strptime(self.config.data.sample_data_start_date, "%Y-%m-%d")
        else:
            start_date = datetime.now() - timedelta(days=days)
        
        logger.info(f"Generating {days} days of sample data ({records_per_day} records/day, {subscription_count} subscriptions)")
        
        # Generate subscriptions
        subscriptions = [self._generate_subscription_id() for _ in range(subscription_count)]
        
        # Generate dates
        dates = self._generate_date_range(start_date, days)
        
        # Prepare data lists
        rows = []
        
        # Generate records for each day
        for date_val in dates:
            # Vary records per day slightly (using Python random to avoid numpy types)
            num_records = int(records_per_day * random.uniform(0.8, 1.2))
            
            for _ in range(num_records):
                # Select category and subcategory (using Python random)
                category = random.choice(list(METER_CATEGORIES.keys()))
                subcategory = random.choice(METER_CATEGORIES[category])
                
                # Generate base cost with trend (using Python random)
                base_cost = random.uniform(10, 1000)
                cost = self._generate_time_series_cost(base_cost, 0.001)
                
                # Generate usage quantity (using Python random)
                quantity = cost / random.uniform(0.01, 1.0)
                effective_price = cost / quantity if quantity > 0 else random.uniform(0.01, 1.0)
                
                # Generate resource details (using Python random)
                region = random.choice(AZURE_REGIONS)
                subscription_id = random.choice(subscriptions)
                tier = random.choice(SERVICE_TIERS)
                unit = random.choice(UNITS[category])
                
                # Create record with REQUIRED and RECOMMENDED columns only
                # Ensure all values are native Python types (not numpy types)
                record = (
                    date_val.date(),  # usage_date - DATE type
                    float(round(cost, 10)),  # cost_in_billing_currency (ensure Python float)
                    float(round(quantity, 6)),  # quantity (ensure Python float)
                    str(category),  # meter_category (ensure Python str)
                    str(region),  # resource_location (ensure Python str)
                    str(subscription_id),  # subscription_id (ensure Python str)
                    float(round(effective_price, 10)),  # effective_price (ensure Python float)
                    str(CURRENCY),  # billing_currency_code (ensure Python str)
                    str(f"{subcategory} - {tier}"),  # plan_name (ensure Python str)
                    str(subcategory),  # meter_sub_category (ensure Python str)
                    str(unit),  # unit_of_measure (ensure Python str)
                )
                
                rows.append(record)
        
        # Define schema
        schema = StructType([
            StructField("usage_date", DateType(), True),
            StructField("cost_in_billing_currency", DoubleType(), True),
            StructField("quantity", DoubleType(), True),
            StructField("meter_category", StringType(), True),
            StructField("resource_location", StringType(), True),
            StructField("subscription_id", StringType(), True),
            StructField("effective_price", DoubleType(), True),
            StructField("billing_currency_code", StringType(), True),
            StructField("plan_name", StringType(), True),
            StructField("meter_sub_category", StringType(), True),
            StructField("unit_of_measure", StringType(), True),
        ])
        
        # Create PySpark DataFrame
        df = spark.createDataFrame(rows, schema=schema)
        
        # Sort by date
        df = df.orderBy("usage_date")
        
        logger.info(f"Generated records from {dates[0].date()} to {dates[-1].date()}")
        
        return df
    
    def load_data(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  **filters) -> DataFrame:
        """
        Load data from Delta table or generate sample data (Section 3.1.1)
        
        If config.data.generate_sample_data is True, generates sample data instead of loading from Delta.
        Otherwise, loads data from the configured Delta table.
        
        Args:
            start_date: Start date filter (YYYY-MM-DD) - only used when loading from Delta
            end_date: End date filter (YYYY-MM-DD) - only used when loading from Delta
            **filters: Additional filters (category, region, etc.)
            
        Returns:
            Spark DataFrame with Azure cost data
        """
        spark = self.validate_spark()
        
        # Check if sample data generation is enabled
        if self.config.data.generate_sample_data:
            logger.info("Sample data generation enabled - generating synthetic data")
            df = self.generate_sample_data()
        else:
            # Load from Delta table (original behavior)
            logger.info(f"Loading data from {self.delta_table_path}")
            # Read from Delta table
            df = spark.read.format("delta").table(self.delta_table_path)

        # Apply filters (using snake_case column names)
        date_col = self.config.feature.date_column
        category = filters.get('category')
        if start_date:
            df = df.filter(col(date_col) >= start_date)
        if end_date:
            df = df.filter(col(date_col) <= end_date)
        if category:
            df = df.filter(col("meter_category") == category)
        
        # logger.info(f"Loaded {df.count()} records")
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
