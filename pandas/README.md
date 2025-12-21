# Azure Cost Management Forecasting - Pandas Implementation

This directory contains the Pandas-based implementation of the Azure Cost Management forecasting project.

## 📁 Directory Structure

```
pandas/
├── README.md
├── requirements.txt
├── notebooks/           # Jupyter notebooks
│   ├── 01_data_generation.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_prophet_model.ipynb
│   ├── 04_arima_model.ipynb
│   ├── 05_xgboost_model.ipynb
│   └── 06_model_comparison.ipynb
├── data/               # Generated data files
│   ├── sample_azure_costs.csv
│   ├── sample_azure_costs_small.csv
│   ├── daily_costs_aggregated.csv
│   └── forecasting_data.pkl
└── utils/              # Utility functions
    └── data_utils.py
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
cd /Users/sabbineni/projects/acm/pandas
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. **Run Notebooks**
```bash
jupyter lab
```

### 3. **Execute in Order**
1. `01_data_generation.ipynb` - Generate sample data
2. `02_data_exploration.ipynb` - Explore and analyze data
3. `03_prophet_model.ipynb` - Prophet forecasting
4. `04_arima_model.ipynb` - ARIMA forecasting
5. `05_xgboost_model.ipynb` - XGBoost forecasting
6. `06_model_comparison.ipynb` - Compare all models

## 📊 Features

- **Data Generation**: 50,000 realistic Azure cost records
- **Regional Focus**: 90% East US, 10% South Central US
- **Currency**: USD only
- **Multiple Models**: Prophet, ARIMA, XGBoost
- **Interactive Visualizations**: Plotly charts and dashboards
- **Model Comparison**: Performance metrics and forecasts

## 🎯 Expected Results

- **Data**: 50K records with $57M+ total cost
- **Forecasts**: 30-90 day predictions with confidence intervals
- **Visualizations**: Interactive charts and model comparisons
- **Performance**: RMSE, MAE, MAPE metrics for all models

## 💡 Best For

- **Single-machine processing**
- **Interactive analysis**
- **Rapid prototyping**
- **Small to medium datasets**
- **Development and testing**

## 🔧 Requirements

- Python 3.9+
- 8GB+ RAM recommended
- 5GB+ free disk space


