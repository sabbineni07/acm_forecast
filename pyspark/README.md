# Azure Cost Management Forecasting - PySpark Implementation

This directory contains the PySpark-based implementation of the Azure Cost Management forecasting project for distributed processing.

## 📁 Directory Structure

```
pyspark/
├── README.md
├── requirements.txt
├── notebooks/           # PySpark scripts
│   ├── 01_data_generation_pyspark.py
│   ├── 02_data_exploration_pyspark.py
│   ├── 03_prophet_model_pyspark.py
│   ├── 04_arima_model_pyspark.py
│   ├── 05_xgboost_model_pyspark.py
│   └── 06_model_comparison_pyspark.py
├── data/               # Generated data files
│   ├── sample_azure_costs.parquet
│   ├── sample_azure_costs.csv
│   ├── daily_costs_aggregated.parquet
│   └── forecasting_data.pkl
└── utils/              # Utility functions
    └── data_utils.py
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
cd /Users/sabbineni/projects/acm/pyspark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. **Run PySpark Scripts**
```bash
# Generate data
python notebooks/01_data_generation_pyspark.py

# Explore data
python notebooks/02_data_exploration_pyspark.py

# Train Prophet model
python notebooks/03_prophet_model_pyspark.py
```

### 3. **Execute All Scripts in Order**
```bash
# Generate data
python notebooks/01_data_generation_pyspark.py

# Explore data
python notebooks/02_data_exploration_pyspark.py

# Train Prophet model
python notebooks/03_prophet_model_pyspark.py

# Train ARIMA model
python notebooks/04_arima_model_pyspark.py

# Train XGBoost model
python notebooks/05_xgboost_model_pyspark.py

# Compare all models
python notebooks/06_model_comparison_pyspark.py
```

### 4. **Or Use Jupyter**
```bash
jupyter lab
# Open and run the .py files as notebooks
```

## 📊 Features

- **Distributed Processing**: Leverages Spark for large-scale data processing
- **Data Generation**: 50,000+ records with parallel processing
- **Regional Focus**: 90% East US, 10% South Central US
- **Currency**: USD only
- **Multiple Models**: Prophet, ARIMA, XGBoost with Spark integration
- **Optimized Storage**: Parquet format for better performance
- **Scalable**: Can handle millions of records

## 🎯 Expected Results

- **Data**: 50K+ records with $57M+ total cost
- **Performance**: Faster processing for large datasets
- **Storage**: Efficient Parquet format
- **Scalability**: Can scale to cluster environments
- **Forecasts**: Same quality as Pandas version but faster

## 💡 Best For

- **Large-scale data processing**
- **Production environments**
- **Cluster computing**
- **Big data scenarios**
- **Performance-critical applications**

## 🔧 Requirements

- Python 3.9+
- Java 8+ (required for Spark)
- 16GB+ RAM recommended
- 10GB+ free disk space
- Spark cluster (optional, can run locally)

## ⚡ Performance Benefits

- **Parallel Processing**: Multiple cores utilization
- **Memory Management**: Efficient Spark memory management
- **Optimized Storage**: Parquet format with compression
- **Lazy Evaluation**: Optimized query execution
- **Caching**: Intelligent DataFrame caching

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   PySpark ETL   │───▶│  Forecasting    │
│   (CSV/Parquet) │    │   (Transform)   │    │   Models        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Results &     │
                       │  Visualizations │
                       └─────────────────┘
```

## 🚀 Running on Databricks

This implementation is optimized for Databricks environments:

1. **Upload notebooks** to Databricks workspace
2. **Install requirements** in cluster libraries
3. **Run notebooks** in sequence
4. **Scale cluster** as needed for larger datasets

## 📈 Scaling Considerations

- **Local Mode**: Single machine, good for development
- **Cluster Mode**: Multiple machines, production ready
- **Memory**: Increase driver and executor memory for large datasets
- **Cores**: More cores = faster processing
- **Storage**: Use distributed storage (HDFS, S3) for very large datasets
