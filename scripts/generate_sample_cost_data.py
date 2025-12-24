#!/usr/bin/env python3
"""
Generate Synthetic Azure Cost Management Sample Data

This script generates realistic synthetic Azure cost data for development and testing.
The data structure matches the REQUIRED schema for ACM Forecast framework using snake_case.

Supports multiple output formats:
- CSV: Simple CSV file
- Parquet: Efficient columnar format
- Delta: Delta table format (requires PySpark) - Can be used with data_source.load_from_delta()

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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import uuid
from pathlib import Path
from typing import List

# Azure regions
AZURE_REGIONS = [
    "East US", "East US 2", "West US", "West US 2", "West US 3",
    "Central US", "South Central US", "North Central US",
    "West Central US", "Canada East", "Canada Central",
    "UK South", "UK West", "West Europe", "North Europe",
    "France Central", "Germany West Central", "Switzerland North",
    "Southeast Asia", "East Asia", "Japan East", "Japan West",
    "Australia East", "Australia Southeast", "Brazil South",
    "South Africa North", "UAE North"
]

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
        Generated cost value
    """
    # Add trend
    cost = base_cost * (1 + trend)
    
    # Add weekly seasonality (lower costs on weekends)
    if seasonality:
        day_of_week = np.random.randint(0, 7)
        if day_of_week >= 5:  # Weekend
            cost *= 0.7
        else:  # Weekday
            cost *= 1.1
    
    # Add random variation
    cost *= np.random.uniform(0.8, 1.2)
    
    # Ensure positive
    return max(cost, 0.01)


def generate_sample_data(
    days: int = 365,
    records_per_day: int = 100,
    subscription_count: int = 3,
    start_date: datetime = None
) -> pd.DataFrame:
    """
    Generate synthetic Azure cost management data with REQUIRED columns only
    
    Args:
        days: Number of days of historical data
        records_per_day: Average number of records per day
        subscription_count: Number of different subscriptions
        start_date: Start date (defaults to days ago from today)
        
    Returns:
        DataFrame with synthetic cost data using snake_case column names
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    # Generate subscriptions
    subscriptions = [generate_subscription_id() for _ in range(subscription_count)]
    
    # Generate dates
    dates = generate_date_range(start_date, days)
    
    # Prepare data lists
    data = []
    
    # Generate records for each day
    for date in dates:
        # Vary records per day slightly
        num_records = int(records_per_day * np.random.uniform(0.8, 1.2))
        
        for _ in range(num_records):
            # Select category and subcategory
            category = np.random.choice(list(METER_CATEGORIES.keys()))
            subcategory = np.random.choice(METER_CATEGORIES[category])
            
            # Generate base cost with trend
            base_cost = np.random.uniform(10, 1000)
            cost = generate_time_series_cost(base_cost, 0.001)
            
            # Generate usage quantity
            quantity = cost / np.random.uniform(0.01, 1.0)
            effective_price = cost / quantity if quantity > 0 else np.random.uniform(0.01, 1.0)
            
            # Generate resource details
            region = np.random.choice(AZURE_REGIONS)
            subscription_id = np.random.choice(subscriptions)
            tier = np.random.choice(SERVICE_TIERS)
            unit = np.random.choice(UNITS[category])
            
            # Create record with REQUIRED and RECOMMENDED columns only
            record = {
                # REQUIRED COLUMNS (Minimum viable set)
                "usage_date": date.date(),  # DATE type
                "cost_in_billing_currency": round(cost, 10),
                "quantity": round(quantity, 6),
                "meter_category": category,
                "resource_location": region,
                
                # RECOMMENDED COLUMNS (Full functionality)
                "subscription_id": subscription_id,
                "effective_price": round(effective_price, 10),
                "billing_currency_code": CURRENCY,
                "plan_name": f"{subcategory} - {tier}",
                
                # OPTIONAL COLUMNS (Nice to have)
                "meter_sub_category": subcategory,
                "unit_of_measure": unit,
            }
            
            data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values("usage_date").reset_index(drop=True)
    
    # Ensure usage_date is DATE type (not datetime)
    df["usage_date"] = pd.to_datetime(df["usage_date"]).dt.date
    
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
    
    # Generate data
    df = generate_sample_data(
        days=args.days,
        records_per_day=args.records_per_day,
        subscription_count=args.subscriptions,
        start_date=start_date
    )
    
    print(f"\nGenerated {len(df):,} records")
    print(f"\nDate range: {df['usage_date'].min()} to {df['usage_date'].max()}")
    print(f"Total cost: ${df['cost_in_billing_currency'].sum():,.2f}")
    print(f"\nCategories: {df['meter_category'].value_counts().to_dict()}")
    print(f"\nRegions: {df['resource_location'].value_counts().head(5).to_dict()}")
    
    # Save data
    print(f"\nSaving to {args.output}...")
    if output_format == "delta":
        # Delta table format - requires PySpark
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.types import (
                StructType, StructField, StringType, DoubleType, DateType
            )
            from delta import configure_spark_with_delta_pip
            
            print("  Creating Spark session with Delta Lake support...")
            builder = SparkSession.builder \
                .appName("GenerateSampleData") \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            # Try to configure with delta-pip if available
            try:
                spark = configure_spark_with_delta_pip(builder).getOrCreate()
            except Exception as e:
                # Fallback to regular Spark if delta-pip not available
                print(f"  Warning: Could not configure Delta Lake, using regular Spark: {e}")
                spark = builder.getOrCreate()
            
            print("  Converting pandas DataFrame to Spark DataFrame...")
            # Convert usage_date to DateType
            df['usage_date'] = pd.to_datetime(df['usage_date']).dt.date
            
            # Create Spark DataFrame with proper schema
            spark_df = spark.createDataFrame(df)
            
            # Write as Delta table
            output_path_str = str(Path(args.output).absolute())
            print(f"  Writing Delta table to: {output_path_str}")
            
            spark_df.write \
                .format("delta") \
                .mode("overwrite") \
                .option("overwriteSchema", "true") \
                .save(output_path_str)
            
            print(f"✅ Successfully saved {len(df):,} records as Delta table to {args.output}")
            print(f"\n📝 To use this Delta table in config:")
            print(f"   Option 1 (local path): delta_table_path: \"{output_path_str}\"")
            print(f"   Option 2 (table name): Create database and register table:")
            print(f"     spark.sql('CREATE DATABASE IF NOT EXISTS test_db')")
            print(f"     spark.sql(f'CREATE TABLE IF NOT EXISTS test_db.sample_costs USING DELTA LOCATION \\\"{output_path_str}\\\"')")
            print(f"     Then use: delta_table_path: \"test_db.sample_costs\"")
            
            spark.stop()
            
        except ImportError as e:
            print(f"\n❌ Error: Delta format requires PySpark and Delta Lake")
            print(f"   Install with: pip install pyspark delta-spark")
            print(f"   Or use CSV/Parquet format instead")
            print(f"\n   Error details: {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Error creating Delta table: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif output_format == "parquet":
        df.to_parquet(args.output, index=False, engine="pyarrow")
        print(f"✅ Successfully saved {len(df):,} records to {args.output}")
    else:
        df.to_csv(args.output, index=False)
        print(f"✅ Successfully saved {len(df):,} records to {args.output}")
    
    if output_format != "delta":
        print(f"\nData preview:")
        print(df.head(10))
        print(f"\nData info:")
        print(df.info())
        print(f"\nColumn names (snake_case):")
        print(list(df.columns))


if __name__ == "__main__":
    main()
