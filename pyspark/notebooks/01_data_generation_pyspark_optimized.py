# Azure Cost Management Data Generation - PySpark Optimized Version
# Using PySpark DataFrames throughout for better performance and scalability

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import random
import numpy as np
from datetime import datetime, timedelta

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("AzureCostDataGeneration") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("PySpark session initialized successfully!")
print(f"Spark version: {spark.version}")
print(f"Available cores: {spark.sparkContext.defaultParallelism}")

# Define sample data for realistic Azure cost data generation
AZURE_REGIONS = ['East US', 'South Central US']
REGION_WEIGHTS = [0.9, 0.1]  # 90% East US, 10% South Central US

METER_CATEGORIES = {
    'Compute': ['Virtual Machines', 'Container Instances', 'App Service', 'Functions', 'Batch'],
    'Storage': ['Blob Storage', 'File Storage', 'Disk Storage', 'Archive Storage', 'Data Lake'],
    'Network': ['Bandwidth', 'Load Balancer', 'VPN Gateway', 'Application Gateway', 'CDN'],
    'Database': ['SQL Database', 'Cosmos DB', 'Redis Cache', 'PostgreSQL', 'MySQL'],
    'Analytics': ['Data Factory', 'Stream Analytics', 'HDInsight', 'Synapse', 'Power BI'],
    'AI/ML': ['Cognitive Services', 'Machine Learning', 'Bot Service', 'Computer Vision', 'Speech Services'],
    'Security': ['Key Vault', 'Security Center', 'Azure AD', 'Sentinel', 'Defender'],
    'Management': ['Monitor', 'Log Analytics', 'Backup', 'Site Recovery', 'Policy']
}

SERVICE_TIERS = ['Basic', 'Standard', 'Premium', 'Free', 'Consumption']
RESOURCE_TYPES = [
    'Microsoft.Compute/virtualMachines',
    'Microsoft.Storage/storageAccounts',
    'Microsoft.Network/loadBalancers',
    'Microsoft.Sql/servers',
    'Microsoft.Web/sites',
    'Microsoft.ContainerService/managedClusters',
    'Microsoft.CognitiveServices/accounts',
    'Microsoft.KeyVault/vaults'
]

UNITS_OF_MEASURE = [
    '1 Hour', '1 GB', '1 GB-Month', '1 GB-Hour', '1 TB', '1 TB-Month',
    '1 Request', '1 Transaction', '1 API Call', '1 Unit', '1 Node',
    '1 Instance', '1 Core', '1 vCPU', '1 GB-Second'
]

print("Sample data definitions created successfully!")

# Generate base DataFrame with required number of records
num_records = 50000
print(f"Generating {num_records:,} records...")

# Create a DataFrame with row numbers
df_base = spark.range(num_records).toDF("id")

# Add random dates within the range
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)
date_range_days = (end_date - start_date).days

# Add random dates and times using PySpark functions
df_with_dates = df_base.withColumn(
    "random_days", 
    (rand() * date_range_days).cast(IntegerType())
).withColumn(
    "random_hours", 
    (rand() * 24).cast(IntegerType())
).withColumn(
    "random_minutes", 
    (rand() * 60).cast(IntegerType())
).withColumn(
    "UsageDateTime", 
    date_add(lit(start_date), col("random_days")) + 
    expr("INTERVAL random_hours HOURS") + 
    expr("INTERVAL random_minutes MINUTES")
)

# Add other required columns using PySpark functions
df_with_basic_cols = df_with_dates.withColumn(
    "SubscriptionGuid", 
    expr("uuid()")
).withColumn(
    "ResourceLocation", 
    when(rand() < 0.9, "East US").otherwise("South Central US")
).withColumn(
    "MeterCategory", 
    element_at(lit(list(METER_CATEGORIES.keys())), (rand() * 8 + 1).cast(IntegerType()))
).withColumn(
    "Currency", 
    lit("USD")
)

# Add more columns based on category using PySpark functions
df_final = df_with_basic_cols.withColumn(
    "MeterSubCategory",
    when(col("MeterCategory") == "Compute", element_at(lit(METER_CATEGORIES['Compute']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "Storage", element_at(lit(METER_CATEGORIES['Storage']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "Network", element_at(lit(METER_CATEGORIES['Network']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "Database", element_at(lit(METER_CATEGORIES['Database']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "Analytics", element_at(lit(METER_CATEGORIES['Analytics']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "AI/ML", element_at(lit(METER_CATEGORIES['AI/ML']), (rand() * 5 + 1).cast(IntegerType())))
    .when(col("MeterCategory") == "Security", element_at(lit(METER_CATEGORIES['Security']), (rand() * 5 + 1).cast(IntegerType())))
    .otherwise(element_at(lit(METER_CATEGORIES['Management']), (rand() * 5 + 1).cast(IntegerType())))
).withColumn(
    "ResourceRate",
    when(col("MeterCategory") == "Compute", rand() * 1.95 + 0.05)
    .when(col("MeterCategory") == "Storage", rand() * 0.099 + 0.001)
    .when(col("MeterCategory") == "Network", rand() * 0.49 + 0.01)
    .when(col("MeterCategory") == "Database", rand() * 4.9 + 0.1)
    .when(col("MeterCategory") == "Analytics", rand() * 0.98 + 0.02)
    .when(col("MeterCategory") == "AI/ML", rand() * 2.99 + 0.01)
    .when(col("MeterCategory") == "Security", rand() * 1.95 + 0.05)
    .otherwise(rand() * 0.49 + 0.01)
).withColumn(
    "UsageQuantity",
    when(col("MeterCategory") == "Compute", rand() * 999 + 1)
    .when(col("MeterCategory") == "Storage", rand() * 9999 + 1)
    .when(col("MeterCategory") == "Network", rand() * 999 + 1)
    .when(col("MeterCategory") == "Database", rand() * 99 + 1)
    .when(col("MeterCategory") == "Analytics", rand() * 999 + 1)
    .when(col("MeterCategory") == "AI/ML", rand() * 9999 + 1)
    .when(col("MeterCategory") == "Security", rand() * 99 + 1)
    .otherwise(rand() * 999 + 1)
)

# Calculate PreTaxCost and add remaining columns
df_azure_costs = df_final.withColumn(
    "PreTaxCost", 
    col("UsageQuantity") * col("ResourceRate")
).withColumn(
    "ResourceGroup", 
    concat(lit("rg-"), lower(col("MeterCategory")), lit("-"), (rand() * 10 + 1).cast(IntegerType()))
).withColumn(
    "MeterId", 
    concat(lit("meter-"), lower(col("MeterCategory")), lit("-"), (rand() * 9000 + 1000).cast(IntegerType()))
).withColumn(
    "MeterName", 
    concat(col("MeterSubCategory"), lit(" - "), col("ResourceLocation"))
).withColumn(
    "MeterRegion", 
    col("ResourceLocation")
).withColumn(
    "ConsumedService", 
    concat(lit("Microsoft."), col("MeterCategory"))
).withColumn(
    "ResourceType", 
    element_at(lit(RESOURCE_TYPES), (rand() * len(RESOURCE_TYPES) + 1).cast(IntegerType()))
).withColumn(
    "InstanceId", 
    concat(lit("instance-"), (rand() * 90000 + 10000).cast(IntegerType()))
).withColumn(
    "Tags", 
    lit('{"Environment": "Production", "Owner": "team-backend", "Project": "project-1", "CostCenter": "CC-100"}')
).withColumn(
    "OfferId", 
    concat(lit("MS-AZR-"), (rand() * 9000 + 1000).cast(IntegerType()))
).withColumn(
    "AdditionalInfo", 
    concat(lit("Additional info for "), col("MeterCategory"))
).withColumn(
    "ServiceInfo1", 
    concat(lit("Service info 1 - "), (rand() * 100 + 1).cast(IntegerType()))
).withColumn(
    "ServiceInfo2", 
    concat(lit("Service info 2 - "), (rand() * 100 + 1).cast(IntegerType()))
).withColumn(
    "ServiceName", 
    concat(lit("Azure "), col("MeterSubCategory"))
).withColumn(
    "ServiceTier", 
    element_at(lit(SERVICE_TIERS), (rand() * len(SERVICE_TIERS) + 1).cast(IntegerType()))
).withColumn(
    "UnitOfMeasure", 
    element_at(lit(UNITS_OF_MEASURE), (rand() * len(UNITS_OF_MEASURE) + 1).cast(IntegerType()))
)

# Apply seasonal and weekend adjustments using PySpark functions
df_with_adjustments = df_azure_costs.withColumn(
    "month", month(col("UsageDateTime"))
).withColumn(
    "day_of_week", dayofweek(col("UsageDateTime"))
).withColumn(
    "seasonal_multiplier", 
    when(col("month").isin([11, 12, 1]), 1.3)
    .when(col("month").isin([6, 7, 8]), 1.1)
    .otherwise(1.0)
).withColumn(
    "weekend_multiplier", 
    when(col("day_of_week").isin([1, 7]), 0.7).otherwise(1.0)
).withColumn(
    "PreTaxCost", 
    col("PreTaxCost") * col("seasonal_multiplier") * col("weekend_multiplier")
)

# Select final columns in the correct order
final_columns = [
    "SubscriptionGuid", "ResourceGroup", "ResourceLocation", "UsageDateTime",
    "MeterCategory", "MeterSubCategory", "MeterId", "MeterName", "MeterRegion",
    "UsageQuantity", "ResourceRate", "PreTaxCost", "ConsumedService",
    "ResourceType", "InstanceId", "Tags", "OfferId", "AdditionalInfo",
    "ServiceInfo1", "ServiceInfo2", "ServiceName", "ServiceTier",
    "Currency", "UnitOfMeasure"
]

df_final_result = df_with_adjustments.select(*final_columns)

# Cache the DataFrame for better performance
df_final_result.cache()

# Display basic statistics using PySpark
print(f"Final DataFrame created with {df_final_result.count()} records")

# Get date range using PySpark
date_range = df_final_result.select(
    min("UsageDateTime").alias("min_date"),
    max("UsageDateTime").alias("max_date")
).collect()[0]

print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")

# Get total cost using PySpark
total_cost = df_final_result.select(sum("PreTaxCost")).collect()[0][0]
print(f"Total cost: ${total_cost:,.2f}")

# Display sample data
print("\nSample data:")
df_final_result.show(5, truncate=False)

# Save data using PySpark (Parquet format is recommended)
output_path_parquet = "/Users/sabbineni/projects/acm/pyspark/data/sample_azure_costs.parquet"
print(f"\nSaving data to Parquet format: {output_path_parquet}")

df_final_result.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(output_path_parquet)

print("✅ Data saved to Parquet format successfully!")

# Also save as CSV for compatibility
output_path_csv = "/Users/sabbineni/projects/acm/pyspark/data/sample_azure_costs.csv"
print(f"Saving data to CSV format: {output_path_csv}")

df_final_result.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path_csv)

print("✅ Data saved to CSV format successfully!")

# Create a smaller sample for quick testing
sample_df = df_final_result.sample(0.1, seed=42)
sample_path = "/Users/sabbineni/projects/acm/pyspark/data/sample_azure_costs_small.parquet"

sample_df.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(sample_path)

print(f"✅ Sample data saved to: {sample_path}")
print(f"Sample size: {sample_df.count()} records")

# Create daily aggregated data for time series analysis using PySpark
print("\nCreating daily aggregated data...")

# Group by date and category using PySpark
daily_costs = df_final_result.groupBy(
    date_trunc("day", col("UsageDateTime")).alias("UsageDateTime"),
    col("MeterCategory")
).agg(
    sum("PreTaxCost").alias("PreTaxCost"),
    sum("UsageQuantity").alias("UsageQuantity")
)

# Add time-based features using PySpark functions
daily_costs_with_features = daily_costs.withColumn(
    "Year", year(col("UsageDateTime"))
).withColumn(
    "Month", month(col("UsageDateTime"))
).withColumn(
    "Day", day(col("UsageDateTime"))
).withColumn(
    "DayOfWeek", dayofweek(col("UsageDateTime"))
).withColumn(
    "DayOfYear", dayofyear(col("UsageDateTime"))
).withColumn(
    "IsWeekend", when(col("DayOfWeek").isin([1, 7]), 1).otherwise(0)
)

# Cache the daily aggregated data
daily_costs_with_features.cache()

# Save daily aggregated data
daily_path = "/Users/sabbineni/projects/acm/pyspark/data/daily_costs_aggregated.parquet"
daily_costs_with_features.write \
    .mode("overwrite") \
    .option("compression", "snappy") \
    .parquet(daily_path)

print(f"✅ Daily aggregated data saved to: {daily_path}")
print(f"Daily records: {daily_costs_with_features.count()}")

# Display sample of daily data
print("\nSample daily aggregated data:")
daily_costs_with_features.show(10, truncate=False)

# Display basic statistics using PySpark
print("\n=== Dataset Overview ===")
print(f"Shape: {df_final_result.count()} records, {len(df_final_result.columns)} columns")
print(f"Columns: {df_final_result.columns}")

# Cost analysis by category using PySpark
print("\n=== Cost Analysis by Meter Category ===")
category_analysis = df_final_result.groupBy("MeterCategory") \
    .agg(
        sum("PreTaxCost").alias("TotalCost"),
        avg("PreTaxCost").alias("AvgCost"),
        count("*").alias("RecordCount"),
        sum("UsageQuantity").alias("TotalUsage"),
        avg("ResourceRate").alias("AvgRate"),
        stddev("PreTaxCost").alias("CostStdDev")
    ) \
    .orderBy(desc("TotalCost"))

category_analysis.show(truncate=False)

# Calculate percentage of total cost using PySpark
total_cost = df_final_result.select(sum("PreTaxCost")).collect()[0][0]
category_analysis_with_pct = category_analysis.withColumn(
    "CostPercentage", 
    (col("TotalCost") / total_cost * 100).cast("decimal(5,2)")
)

print(f"\nTotal Cost: ${total_cost:,.2f}")
print("\nCost Distribution by Category:")
category_analysis_with_pct.select("MeterCategory", "TotalCost", "CostPercentage").show(truncate=False)

# Region distribution using PySpark
print("\n=== Region Distribution ===")
region_dist = df_final_result.groupBy("ResourceLocation") \
    .agg(
        count("*").alias("RecordCount"),
        sum("PreTaxCost").alias("TotalCost")
    ) \
    .withColumn(
        "Percentage", 
        (col("RecordCount") / df_final_result.count() * 100).cast("decimal(5,2)")
    )

region_dist.show(truncate=False)

print("\n✅ Azure cost data generation completed successfully!")
print("📊 Data is ready for PySpark-based analysis and forecasting")
print("🚀 All operations performed using PySpark DataFrames for optimal performance")

# Display final summary
print("\n=== Final Summary ===")
print(f"Total records generated: {df_final_result.count():,}")
print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
print(f"Total cost: ${total_cost:,.2f}")
print(f"Categories: {df_final_result.select('MeterCategory').distinct().count()}")
print(f"Regions: {df_final_result.select('ResourceLocation').distinct().count()}")
print(f"Currency: {df_final_result.select('Currency').distinct().collect()[0][0]}")

spark.stop()


