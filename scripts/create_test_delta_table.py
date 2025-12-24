#!/usr/bin/env python3
"""
Create Delta Table from Sample CSV Data

This script converts the generated CSV sample data into a Delta table format
that can be used by data_source.load_from_delta().

Usage:
    # First generate CSV data
    python scripts/generate_sample_cost_data.py --output data/sample_costs.csv --days 365
    
    # Then convert to Delta table
    python scripts/create_test_delta_table.py \
        --input data/sample_costs.csv \
        --output data/delta/sample_costs \
        --database test_db \
        --table sample_costs

After running, you can use in config.yaml:
    data:
      delta_table_path: "test_db.sample_costs"
      database_name: "test_db"
      table_name: "sample_costs"
"""

import argparse
import pandas as pd
from pathlib import Path
import sys


def create_delta_table(input_csv: str, output_path: str, database: str = None, table: str = None):
    """
    Convert CSV file to Delta table
    
    Args:
        input_csv: Path to input CSV file
        output_path: Path to output Delta table directory
        database: Optional database name to register table
        table: Optional table name to register
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType
    except ImportError:
        print("❌ Error: PySpark is required. Install with: pip install pyspark")
        return 1
    
    try:
        from delta import configure_spark_with_delta_pip
        delta_available = True
    except ImportError:
        print("⚠️  Warning: delta-spark not available. Trying without Delta Lake configuration...")
        delta_available = False
    
    print(f"Loading CSV data from {input_csv}...")
    df_pandas = pd.read_csv(input_csv)
    
    # Convert usage_date to date type
    df_pandas['usage_date'] = pd.to_datetime(df_pandas['usage_date']).dt.date
    
    print(f"  Loaded {len(df_pandas):,} records")
    print(f"  Columns: {list(df_pandas.columns)}")
    
    # Create Spark session
    print("\nCreating Spark session...")
    builder = SparkSession.builder \
        .appName("CreateTestDeltaTable") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    if delta_available:
        try:
            spark = configure_spark_with_delta_pip(builder).getOrCreate()
        except Exception as e:
            print(f"⚠️  Warning: Could not configure Delta Lake, using regular Spark: {e}")
            spark = builder.getOrCreate()
    else:
        spark = builder.getOrCreate()
    
    # Convert pandas DataFrame to Spark DataFrame
    print("Converting to Spark DataFrame...")
    spark_df = spark.createDataFrame(df_pandas)
    
    # Show schema
    print("\nSpark DataFrame Schema:")
    spark_df.printSchema()
    
    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    output_path_str = str(output_path_obj.absolute())
    
    # Write as Delta table
    print(f"\nWriting Delta table to: {output_path_str}")
    spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save(output_path_str)
    
    print(f"✅ Successfully created Delta table with {len(df_pandas):,} records")
    
    # Optionally register as table in database
    if database and table:
        print(f"\nRegistering table in database...")
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
        
        # Register table
        table_path = f"delta.`{output_path_str}`"
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {database}.{table} 
            USING DELTA 
            LOCATION '{output_path_str}'
        """)
        
        print(f"✅ Table registered as: {database}.{table}")
        print(f"\n📝 Use in config.yaml:")
        print(f"   data:")
        print(f"     delta_table_path: \"{database}.{table}\"")
        print(f"     database_name: \"{database}\"")
        print(f"     table_name: \"{table}\"")
    else:
        print(f"\n📝 To use this Delta table, you can:")
        print(f"   Option 1: Use full path in config")
        print(f"     delta_table_path: \"{output_path_str}\"")
        print(f"   Option 2: Register as table (run with --database and --table flags)")
    
    spark.stop()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV sample data to Delta table format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CSV to Delta table
  python scripts/create_test_delta_table.py \\
      --input data/sample_costs.csv \\
      --output data/delta/sample_costs

  # Convert and register in database
  python scripts/create_test_delta_table.py \\
      --input data/sample_costs.csv \\
      --output data/delta/sample_costs \\
      --database test_db \\
      --table sample_costs
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output Delta table path (directory)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Optional: Database name to register table"
    )
    parser.add_argument(
        "--table",
        type=str,
        default=None,
        help="Optional: Table name to register (requires --database)"
    )
    
    args = parser.parse_args()
    
    if args.table and not args.database:
        print("❌ Error: --table requires --database")
        return 1
    
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    return create_delta_table(args.input, args.output, args.database, args.table)


if __name__ == "__main__":
    sys.exit(main())

