# Azure Cost Management Data Exploration - PySpark Optimized Version
# Using PySpark DataFrames throughout for better performance and scalability

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("AzureCostDataExploration") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("PySpark session initialized successfully!")
print(f"Spark version: {spark.version}")

# Load the data using PySpark
print("Loading Azure cost data...")

# Load main dataset
df = spark.read.parquet("/Users/sabbineni/projects/acm/pyspark/data/sample_azure_costs.parquet")
df = df.withColumn("UsageDateTime", col("UsageDateTime").cast("timestamp"))

# Load daily aggregated data
daily_df = spark.read.parquet("/Users/sabbineni/projects/acm/pyspark/data/daily_costs_aggregated.parquet")
daily_df = daily_df.withColumn("UsageDateTime", col("UsageDateTime").cast("timestamp"))

print(f"Main dataset shape: {df.count()} records, {len(df.columns)} columns")
print(f"Daily aggregated dataset shape: {daily_df.count()} records, {len(daily_df.columns)} columns")

# Get date range using PySpark
date_range = df.select(
    min("UsageDateTime").alias("min_date"),
    max("UsageDateTime").alias("max_date")
).collect()[0]

print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")

# Cache DataFrames for better performance
df.cache()
daily_df.cache()

# Cost Analysis by Category using PySpark
print("\n=== Cost Analysis by Meter Category ===")

category_analysis = df.groupBy("MeterCategory") \
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
total_cost = df.select(sum("PreTaxCost")).collect()[0][0]
category_analysis_with_pct = category_analysis.withColumn(
    "CostPercentage", 
    (col("TotalCost") / total_cost * 100).cast("decimal(5,2)")
)

print(f"\nTotal Cost: ${total_cost:,.2f}")
print("\nCost Distribution by Category:")
category_analysis_with_pct.select("MeterCategory", "TotalCost", "CostPercentage").show(truncate=False)

# Time Series Analysis using PySpark
print("\n=== Time Series Analysis ===")

# Create monthly aggregated data using PySpark
monthly_costs = df.groupBy(
    date_trunc("month", col("UsageDateTime")).alias("UsageDateTime")
).agg(
    sum("PreTaxCost").alias("PreTaxCost"),
    sum("UsageQuantity").alias("UsageQuantity")
).orderBy("UsageDateTime")

print("Monthly Cost Summary:")
monthly_costs.show(truncate=False)

# Calculate growth rates using window functions
window_spec = Window.orderBy("UsageDateTime")
monthly_costs_with_growth = monthly_costs.withColumn(
    "Cost_Growth", 
    ((col("PreTaxCost") - lag("PreTaxCost", 1).over(window_spec)) / 
     lag("PreTaxCost", 1).over(window_spec) * 100).cast("decimal(5,2)")
).withColumn(
    "Usage_Growth",
    ((col("UsageQuantity") - lag("UsageQuantity", 1).over(window_spec)) / 
     lag("UsageQuantity", 1).over(window_spec) * 100).cast("decimal(5,2)")
)

# Get average monthly cost using PySpark
avg_monthly_cost = monthly_costs.select(avg("PreTaxCost")).collect()[0][0]
cost_volatility = monthly_costs.select(stddev("PreTaxCost")).collect()[0][0]

print(f"\nAverage monthly cost: ${avg_monthly_cost:,.2f}")
print(f"Cost volatility (std): ${cost_volatility:,.2f}")

# Seasonal Analysis using PySpark
print("\n=== Seasonal Analysis ===")

# Analyze patterns by month using PySpark
monthly_patterns = df.groupBy(month(col("UsageDateTime")).alias("Month")) \
    .agg(
        sum("PreTaxCost").alias("TotalCost"),
        avg("PreTaxCost").alias("AvgCost"),
        count("*").alias("RecordCount")
    ) \
    .orderBy("Month")

print("Monthly Patterns:")
monthly_patterns.show(truncate=False)

# Analyze patterns by day of week using PySpark
weekday_patterns = df.groupBy(dayofweek(col("UsageDateTime")).alias("DayOfWeek")) \
    .agg(
        sum("PreTaxCost").alias("TotalCost"),
        avg("PreTaxCost").alias("AvgCost"),
        count("*").alias("RecordCount")
    ) \
    .orderBy("DayOfWeek")

print("\nWeekday Patterns:")
weekday_patterns.show(truncate=False)

# Hourly patterns using PySpark
hourly_patterns = df.groupBy(hour(col("UsageDateTime")).alias("Hour")) \
    .agg(avg("PreTaxCost").alias("AvgCost")) \
    .orderBy("Hour")

print("\nHourly Patterns:")
hourly_patterns.show(truncate=False)

# Region Analysis using PySpark
print("\n=== Region Analysis ===")

region_analysis = df.groupBy("ResourceLocation") \
    .agg(
        count("*").alias("RecordCount"),
        sum("PreTaxCost").alias("TotalCost"),
        avg("PreTaxCost").alias("AvgCost")
    ) \
    .withColumn(
        "Percentage", 
        (col("RecordCount") / df.count() * 100).cast("decimal(5,2)")
    ) \
    .orderBy(desc("RecordCount"))

print("Region Distribution:")
region_analysis.show(truncate=False)

# Data Quality Assessment using PySpark
print("\n=== Data Quality Assessment ===")

# Check for missing values using PySpark
print("Missing Values:")
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull()).count()
    if null_count > 0:
        print(f"  {col_name}: {null_count}")

# Check for duplicates using PySpark
duplicates = df.count() - df.dropDuplicates().count()
print(f"\nDuplicate records: {duplicates}")

# Check data consistency using PySpark
print(f"\nData Consistency Checks:")
print(f"Negative costs: {df.filter(col('PreTaxCost') < 0).count()}")
print(f"Zero costs: {df.filter(col('PreTaxCost') == 0).count()}")
print(f"Negative usage: {df.filter(col('UsageQuantity') < 0).count()}")
print(f"Zero usage: {df.filter(col('UsageQuantity') == 0).count()}")

# Summary statistics using PySpark
print(f"\n=== Summary Statistics ===")
print(f"Total records: {df.count():,}")
print(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
print(f"Total cost: ${total_cost:,.2f}")

avg_cost = df.select(avg("PreTaxCost")).collect()[0][0]
print(f"Average cost per record: ${avg_cost:.4f}")

# Calculate median using PySpark (approximate)
median_cost = df.select(expr("percentile_approx(PreTaxCost, 0.5)")).collect()[0][0]
print(f"Median cost per record: ${median_cost:.4f}")

cost_std = df.select(stddev("PreTaxCost")).collect()[0][0]
print(f"Cost standard deviation: ${cost_std:.2f}")

# Prepare data for forecasting models using PySpark
print("\n=== Preparing Data for Forecasting Models ===")

# Create time series data for each category using PySpark
forecasting_data = {}

for category_row in df.select("MeterCategory").distinct().collect():
    category = category_row["MeterCategory"]
    
    # Filter data for this category using PySpark
    category_data = daily_df.filter(col("MeterCategory") == category) \
        .select("UsageDateTime", "PreTaxCost") \
        .orderBy("UsageDateTime")
    
    # Convert to Pandas only for forecasting libraries (they expect Pandas)
    category_pandas = category_data.toPandas()
    if len(category_pandas) > 0:
        category_pandas.set_index("UsageDateTime", inplace=True)
        forecasting_data[category] = category_pandas["PreTaxCost"]
        
        print(f"{category}: {len(category_pandas)} data points, "
              f"Date range: {category_pandas.index.min()} to {category_pandas.index.max()}, "
              f"Total cost: ${category_pandas['PreTaxCost'].sum():,.2f}")

# Create a combined time series for total costs using PySpark
total_daily_costs = daily_df.groupBy("UsageDateTime") \
    .agg(sum("PreTaxCost").alias("PreTaxCost")) \
    .orderBy("UsageDateTime")

total_pandas = total_daily_costs.toPandas()
if len(total_pandas) > 0:
    total_pandas.set_index("UsageDateTime", inplace=True)
    forecasting_data['Total'] = total_pandas['PreTaxCost']
    
    print(f"\nTotal daily costs: {len(total_pandas)} data points")
    print(f"Date range: {total_pandas.index.min()} to {total_pandas.index.max()}")
    print(f"Total cost: ${total_pandas['PreTaxCost'].sum():,.2f}")

# Save forecasting data
import pickle
forecasting_path = "/Users/sabbineni/projects/acm/pyspark/data/forecasting_data.pkl"
with open(forecasting_path, 'wb') as f:
    pickle.dump(forecasting_data, f)

print(f"\nForecasting data saved to: {forecasting_path}")

# Create visualizations using PySpark DataFrames converted to Pandas for plotting
print("\n=== Creating Visualizations ===")

# Convert key DataFrames to Pandas for visualization
category_pandas = category_analysis_with_pct.toPandas()
monthly_pandas = monthly_costs.toPandas()
region_pandas = region_analysis.toPandas()

# Set up plotting
plt.style.use('seaborn-v0_8')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Cost distribution by category (bar chart)
category_pandas.plot(x='MeterCategory', y='TotalCost', kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Total Cost by Category')
ax1.set_xlabel('Meter Category')
ax1.set_ylabel('Total Cost ($)')
ax1.tick_params(axis='x', rotation=45)

# 2. Monthly cost trends
monthly_pandas['UsageDateTime'] = pd.to_datetime(monthly_pandas['UsageDateTime'])
monthly_pandas.plot(x='UsageDateTime', y='PreTaxCost', ax=ax2, color='blue', linewidth=2)
ax2.set_title('Monthly Cost Trends')
ax2.set_xlabel('Date')
ax2.set_ylabel('Total Cost ($)')

# 3. Region distribution (pie chart)
region_pandas.plot(x='ResourceLocation', y='Percentage', kind='pie', ax=ax3, autopct='%1.1f%%')
ax3.set_title('Cost Distribution by Region')
ax3.set_ylabel('')

# 4. Record count by category
category_pandas.plot(x='MeterCategory', y='RecordCount', kind='bar', ax=ax4, color='lightcoral')
ax4.set_title('Record Count by Category')
ax4.set_xlabel('Meter Category')
ax4.set_ylabel('Number of Records')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Create interactive visualizations using Plotly
print("\n=== Creating Interactive Visualizations ===")

# Create comprehensive visualization using PySpark data
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Monthly Cost Trends', 'Cost Distribution by Category', 
                   'Daily Cost Patterns', 'Resource Usage vs Cost'),
    specs=[[{"secondary_y": False}, {"type": "domain"}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Monthly Cost Trends
monthly_pandas['UsageDateTime'] = pd.to_datetime(monthly_pandas['UsageDateTime'])
fig.add_trace(
    go.Scatter(x=monthly_pandas['UsageDateTime'], y=monthly_pandas['PreTaxCost'],
               mode='lines+markers', name='Monthly Cost', line=dict(color='blue')),
    row=1, col=1
)

# 2. Cost Distribution by Category (Pie Chart)
fig.add_trace(
    go.Pie(labels=category_pandas['MeterCategory'], values=category_pandas['TotalCost'],
           name='Cost by Category'),
    row=1, col=2
)

# 3. Daily Cost Patterns
daily_pandas = daily_df.toPandas()
daily_pandas['UsageDateTime'] = pd.to_datetime(daily_pandas['UsageDateTime'])
daily_total = daily_pandas.groupby('UsageDateTime')['PreTaxCost'].sum().reset_index()

fig.add_trace(
    go.Scatter(x=daily_total['UsageDateTime'], y=daily_total['PreTaxCost'],
               mode='lines', name='Daily Cost', line=dict(color='green')),
    row=2, col=1
)

# 4. Resource Usage vs Cost
sample_df = df.sample(0.1, seed=42).toPandas()  # Sample for performance
fig.add_trace(
    go.Scatter(x=sample_df['UsageQuantity'], y=sample_df['PreTaxCost'],
               mode='markers', name='Usage vs Cost', 
               marker=dict(color='red', size=4, opacity=0.6)),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Azure Cost Management - Data Exploration Dashboard",
    showlegend=True
)

fig.show()

print("\n✅ Data exploration completed successfully!")
print("📊 Data is ready for PySpark-based forecasting model development")
print("🚀 All analysis performed using PySpark DataFrames for optimal performance")

# Display sample data
print("\n=== Sample Data ===")
df.show(5, truncate=False)

print("\n=== Data Types ===")
df.printSchema()

# Create summary report
print("\n=== Data Exploration Summary ===")
print(f"✅ Total records analyzed: {df.count():,}")
print(f"✅ Date range: {date_range['min_date']} to {date_range['max_date']}")
print(f"✅ Total cost: ${total_cost:,.2f}")
print(f"✅ Categories: {df.select('MeterCategory').distinct().count()}")
print(f"✅ Regions: {df.select('ResourceLocation').distinct().count()}")
print(f"✅ Data quality: {((df.count() - duplicates) / df.count() * 100):.1f}% clean records")
print(f"✅ Forecasting data prepared for {len(forecasting_data)} categories")

spark.stop()


