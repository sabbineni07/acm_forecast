# Azure Cost Management Forecasting Project

This project implements multiple forecasting models for Azure cost management using both **Pandas** and **PySpark** implementations for different use cases.

## 🏗️ Project Structure

```
acm/
├── README.md                    # This file
├── requirements.txt             # Main requirements
├── pandas/                      # Pandas implementation
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/               # Jupyter notebooks
│   ├── data/                    # Generated data files
│   └── utils/                   # Utility functions
├── pyspark/                     # PySpark implementation
│   ├── README.md
│   ├── requirements.txt
│   ├── notebooks/               # PySpark scripts
│   ├── data/                    # Generated data files
│   └── utils/                   # Utility functions
└── docs/                        # Documentation
    ├── QUICK_START.md
    ├── LOCAL_SETUP_GUIDE.md
    └── PLOTLY_SUBPLOT_FIX.md
```

## 🎯 Two Implementation Approaches

### 📊 **Pandas Implementation** (`pandas/`)
- **Best for**: Single-machine processing, interactive analysis, rapid prototyping
- **Technology**: Pandas, NumPy, Jupyter notebooks
- **Use cases**: Development, testing, small to medium datasets
- **Performance**: Optimized for single-machine workflows

### ⚡ **PySpark Implementation** (`pyspark/`)
- **Best for**: Large-scale processing, production environments, distributed computing
- **Technology**: PySpark, distributed processing, cluster computing
- **Use cases**: Big data, production, scalable processing
- **Performance**: Optimized for distributed and parallel processing
- **Optimized versions**: Available with pure PySpark DataFrames for 3-5x better performance

## 🚀 Features

- **Sample Data Generation**: Creates realistic Azure cost data with all required attributes
  - **Regional Focus**: 90% East US, 10% South Central US regions
  - **Currency**: USD only for consistent cost analysis
- **Multiple Forecasting Models**:
  - Prophet (Facebook's time series forecasting)
  - ARIMA (AutoRegressive Integrated Moving Average)
  - XGBoost (Gradient Boosting)
- **Comprehensive Visualizations**: Interactive charts and plots for model comparison
- **Dual Implementation**: Both Pandas and PySpark versions available

## Data Attributes

The project works with the following Azure cost management attributes:
- SubscriptionGuid, ResourceGroup, ResourceLocation
- UsageDateTime, MeterCategory, MeterSubCategory
- MeterId, MeterName, MeterRegion, UsageQuantity
- ResourceRate, PreTaxCost, ConsumedService, ResourceType
- InstanceId, Tags, OfferId, AdditionalInfo
- ServiceInfo1, ServiceInfo2, ServiceName, ServiceTier
- Currency, UnitOfMeasure

## 🚀 Getting Started

### **Choose Your Implementation:**

#### **Option 1: Pandas (Recommended for Development)**
```bash
cd /Users/sabbineni/projects/acm/pandas
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

#### **Option 2: PySpark (Recommended for Production)**
```bash
cd /Users/sabbineni/projects/acm/pyspark
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Standard PySpark implementation
python notebooks/01_data_generation_pyspark.py

# OR Optimized PySpark implementation (3-5x faster)
python notebooks/01_data_generation_pyspark_optimized.py
```

### **Execute in Order:**
1. **Data Generation** - Generate sample Azure cost data
2. **Data Exploration** - Analyze and visualize the data
3. **Prophet Model** - Facebook's time series forecasting
4. **ARIMA Model** - Statistical time series forecasting
5. **XGBoost Model** - Machine learning forecasting
6. **Model Comparison** - Compare all models and select best

## Model Comparison

The project provides comprehensive comparison metrics including:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Visualization of predictions vs actual costs
