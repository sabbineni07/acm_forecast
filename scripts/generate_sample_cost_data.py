#!/usr/bin/env python3
"""
Generate Synthetic Azure Cost Management Sample Data

This script generates realistic synthetic Azure cost data for development and testing.
The data structure matches the REQUIRED schema for ACM Forecast framework using snake_case.

Supports multiple output formats (all using PySpark):
- CSV: Simple CSV file (written via PySpark)
- Parquet: Efficient columnar format (written via PySpark)
- Delta: Delta table format (requires PySpark and Delta Lake) - Can be used with data_source.load_from_delta()

Note: This script now uses PySpark DataFrames throughout, eliminating the need for pandas.
Data generation creates a PySpark DataFrame directly, which is more efficient for large datasets.

Required columns (minimum):
- usage_date (DATE)
- cost_in_billing_currency (DECIMAL)
- quantity (DECIMAL)
- meter_category (STRING)
- resource_location (STRING)

Recommended columns (full functionality):
- subscription_id (STRING)
- effective_price (DECIMAL)
- billing_currency_code (STRING)
- plan_name (STRING)

Usage:
    # Generate CSV
    python scripts/generate_sample_cost_data.py --output data/sample_costs.csv --days 365
    
    # Generate Parquet
    python scripts/generate_sample_cost_data.py --output data/sample_costs.parquet --format parquet --days 180
    
    # Generate Delta table (requires PySpark)
    python scripts/generate_sample_cost_data.py --output data/delta/sample_costs --format delta --days 365
    # Then use in config: delta_table_path: "default.sample_costs" or full path
"""

import argparse
from datetime import datetime, timedelta
import uuid
from pathlib import Path
from typing import List, Optional
import random

# Azure regions
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


def generate_subscription_id() -> str:
    """Generate a synthetic Azure subscription ID (GUID format)"""
    return str(uuid.uuid4())


def generate_date_range(start_date: datetime, days: int) -> List[datetime]:
    """Generate a list of dates"""
    return [start_date + timedelta(days=i) for i in range(days)]


def generate_time_series_cost(base_cost: float, trend: float, seasonality: bool = True) -> float:
    """
    Generate realistic cost with trend and seasonality
    
    Args:
        base_cost: Base cost value
        trend: Trend coefficient (e.g., 0.001 for 0.1% growth per day)
        seasonality: Add weekly seasonality
        
    Returns:
        Generated cost value (as Python float)
    """
    import random
    
    # Add trend
    cost = base_cost * (1 + trend)
    
    # Add weekly seasonality (lower costs on weekends)
    if seasonality:
        day_of_week = random.randint(0, 6)
        if day_of_week >= 5:  # Weekend
            cost *= 0.7
        else:  # Weekday
            cost *= 1.1
    
    # Add random variation (using Python's random to avoid numpy types)
    cost *= random.uniform(0.8, 1.2)
    
    # Ensure positive and convert to Python float
    return float(max(cost, 0.01))


def generate_sample_data(
    spark_session,
    days: int = 365,
    records_per_day: int = 100,
    subscription_count: int = 3,
    start_date: Optional[datetime] = None
):
    """
    Generate synthetic Azure cost management data with REQUIRED columns only
    
    Args:
        spark_session: PySpark SparkSession instance
        days: Number of days of historical data
        records_per_day: Average number of records per day
        subscription_count: Number of different subscriptions
        start_date: Start date (defaults to days ago from today)
        
    Returns:
        PySpark DataFrame with synthetic cost data using snake_case column names.
        The DataFrame has the following schema:
        - usage_date (DateType)
        - cost_in_billing_currency (DoubleType)
        - quantity (DoubleType)
        - meter_category (StringType)
        - resource_location (StringType)
        - subscription_id (StringType)
        - effective_price (DoubleType)
        - billing_currency_code (StringType)
        - plan_name (StringType)
        - meter_sub_category (StringType)
        - unit_of_measure (StringType)
    """
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
    from pyspark.sql.functions import to_date
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate subscriptions
    subscriptions = [generate_subscription_id() for _ in range(subscription_count)]
    
    # Generate dates
    dates = generate_date_range(start_date, days)
    
    # Prepare data lists
    rows = []
    
    # Generate records for each day
    for date in dates:
        # Vary records per day slightly (using Python random to avoid numpy types)
        num_records = int(records_per_day * random.uniform(0.8, 1.2))
        
        for _ in range(num_records):
            # Select category and subcategory (using Python random)
            category = random.choice(list(METER_CATEGORIES.keys()))
            subcategory = random.choice(METER_CATEGORIES[category])
            
            # Generate base cost with trend (using Python random)
            base_cost = random.uniform(10, 1000)
            cost = generate_time_series_cost(base_cost, 0.001)
            
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
                date.date(),  # usage_date - DATE type
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
    df = spark_session.createDataFrame(rows, schema=schema)
    
    # Sort by date
    df = df.orderBy("usage_date")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Azure Cost Management sample data (snake_case schema)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 365 days of CSV data (required columns only)
  python scripts/generate_sample_cost_data.py --output data/sample_costs.csv --days 365
  
  # Generate 180 days of Parquet data with more records per day
  python scripts/generate_sample_cost_data.py --output data/sample_costs.parquet --days 180 --records-per-day 200 --format parquet
  
  # Generate data for 5 subscriptions
  python scripts/generate_sample_cost_data.py --output data/sample_costs.csv --days 90 --subscriptions 5

Required columns generated:
  - usage_date (DATE)
  - cost_in_billing_currency (DECIMAL)
  - quantity (DECIMAL)
  - meter_category (STRING)
  - resource_location (STRING)

Recommended columns also included:
  - subscription_id (STRING)
  - effective_price (DECIMAL)
  - billing_currency_code (STRING)
  - plan_name (STRING)
        """
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_azure_costs.csv",
        help="Output file path (default: data/sample_azure_costs.csv)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data (default: 365)"
    )
    parser.add_argument(
        "--records-per-day",
        type=int,
        default=100,
        help="Average number of records per day (default: 100)"
    )
    parser.add_argument(
        "--subscriptions",
        type=int,
        default=3,
        help="Number of different subscriptions (default: 3)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "delta"],
        default=None,
        help="Output format (default: inferred from file extension). Use 'delta' for Delta table format."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, defaults to --days ago from today)"
    )
    
    args = parser.parse_args()
    
    # Determine output format
    if args.format is None:
        output_path = Path(args.output)
        if output_path.suffix == ".parquet":
            output_format = "parquet"
        elif output_path.suffix in [".delta", ""] or output_path.is_dir():
            # Delta tables are directories
            output_format = "delta"
        else:
            output_format = "csv"
    else:
        output_format = args.format
    
    # Parse start date
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.days} days of synthetic Azure cost data...")
    print(f"  Records per day: ~{args.records_per_day}")
    print(f"  Subscriptions: {args.subscriptions}")
    print(f"  Expected total records: ~{args.days * args.records_per_day:,}")
    print(f"\nColumns (snake_case):")
    print(f"  Required: usage_date, cost_in_billing_currency, quantity, meter_category, resource_location")
    print(f"  Recommended: subscription_id, effective_price, billing_currency_code, plan_name")
    
    # Initialize Spark session (required for PySpark DataFrame)
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import min as spark_min, max as spark_max, sum as spark_sum, count, col
        
        print("\n  Initializing Spark session...")
        # Configure Spark based on output format
        builder = SparkSession.builder \
            .appName("GenerateSampleData") \
            .config("spark.sql.adaptive.enabled", "true")
        
        # Add Delta Lake support if needed
        if output_format == "delta":
            try:
                from delta import configure_spark_with_delta_pip
                print("  Adding Delta Lake support...")
                spark = configure_spark_with_delta_pip(builder).getOrCreate()
            except ImportError:
                print("  Warning: Delta Lake not available, using regular Spark")
                spark = builder.getOrCreate()
        else:
            spark = builder.getOrCreate()
        
        # Generate data (returns PySpark DataFrame)
        df = generate_sample_data(
            spark_session=spark,
            days=args.days,
            records_per_day=args.records_per_day,
            subscription_count=args.subscriptions,
            start_date=start_date
        )
        
        # Get statistics from PySpark DataFrame
        record_count = df.count()
        date_range = df.agg(
            spark_min("usage_date").alias("min_date"),
            spark_max("usage_date").alias("max_date")
        ).collect()[0]
        total_cost = df.agg(spark_sum("cost_in_billing_currency").alias("total_cost")).collect()[0]["total_cost"]
        
        print(f"\nGenerated {record_count:,} records")
        print(f"\nDate range: {date_range['min_date']} to {date_range['max_date']}")
        print(f"Total cost: ${total_cost:,.2f}")
        
        # Category counts
        category_counts = df.groupBy("meter_category").agg(count("*").alias("count")).collect()
        print(f"\nCategories:")
        for row in category_counts:
            print(f"  {row['meter_category']}: {row['count']:,}")
        
        # Region counts (top 5)
        region_counts = df.groupBy("resource_location").agg(count("*").alias("count")).orderBy(col("count").desc()).limit(5).collect()
        print(f"\nTop Regions:")
        for row in region_counts:
            print(f"  {row['resource_location']}: {row['count']:,}")
        
    except ImportError as e:
        print(f"\n❌ Error: PySpark is required")
        print(f"   Install with: pip install pyspark")
        print(f"   Error details: {e}")
        return 1
    
    # Save data
    print(f"\nSaving to {args.output}...")
    try:
        if output_format == "delta":
            # Delta table format
            output_path_str = str(Path(args.output).absolute())
            print(f"  Writing Delta table to: {output_path_str}")
            
            df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(output_path_str)
            
            record_count = df.count()
            print(f"✅ Successfully saved {record_count:,} records as Delta table to {args.output}")
            print(f"\n📝 To use this Delta table in config:")
            print(f"   Option 1 (local path): delta_table_path: \"{output_path_str}\"")
            print(f"   Option 2 (table name): Create database and register table:")
            print(f"     spark.sql('CREATE DATABASE IF NOT EXISTS test_db')")
            print(f"     spark.sql(f'CREATE TABLE IF NOT EXISTS test_db.sample_costs USING DELTA LOCATION \\\"{output_path_str}\\\"')")
            print(f"     Then use: delta_table_path: \"test_db.sample_costs\"")
            
        elif output_format == "parquet":
            # Write as Parquet using PySpark
            print(f"  Writing Parquet file...")
            df.coalesce(1).write.mode("overwrite").parquet(args.output)
            record_count = df.count()
            print(f"✅ Successfully saved {record_count:,} records to {args.output}")
        else:
            # Write as CSV using PySpark
            print(f"  Writing CSV file...")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(args.output)
            record_count = df.count()
            print(f"✅ Successfully saved {record_count:,} records to {args.output}")
        
        # Show preview for non-Delta formats
        if output_format != "delta":
            print(f"\nData preview (first 10 rows):")
            df.show(10, truncate=False)
            print(f"\nSchema:")
            df.printSchema()
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        print(f"\n❌ Error saving data: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        return 1


if __name__ == "__main__":
    main()
