# 🚀 Azure Cost Management Forecasting - Local Setup Guide

This guide will help you run and test the Azure Cost Management forecasting project on your local laptop.

## 📋 Prerequisites

- **Python 3.9+** (You have Python 3.9.13 ✅)
- **macOS/Linux/Windows** (You're on macOS ✅)
- **8GB+ RAM** (recommended for large datasets)
- **5GB+ free disk space**

## 🛠️ Setup Instructions

### 1. **Environment Setup**

```bash
# Navigate to project directory
cd /Users/sabbineni/projects/acm

# Activate virtual environment
source venv/bin/activate

# Verify installation
python test_installation_simple.py
```

### 2. **Optional: Install XGBoost Support**

If you want to use XGBoost (optional), install OpenMP:

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenMP
brew install libomp
```

## 🎯 Running the Project

### **Method 1: Jupyter Notebooks (Recommended)**

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

**Then open notebooks in order:**
1. `notebooks/01_data_generation.ipynb` - Generate sample data
2. `notebooks/02_data_exploration.ipynb` - Explore and analyze data
3. `notebooks/03_prophet_model.ipynb` - Prophet forecasting
4. `notebooks/04_arima_model.ipynb` - ARIMA forecasting
5. `notebooks/05_xgboost_model.ipynb` - XGBoost forecasting (optional)
6. `notebooks/06_model_comparison.ipynb` - Compare all models

### **Method 2: Python Scripts**

```bash
# Activate environment
source venv/bin/activate

# Run individual notebooks as scripts
jupyter nbconvert --to script --execute notebooks/01_data_generation.ipynb
jupyter nbconvert --to script --execute notebooks/02_data_exploration.ipynb
# ... and so on
```

### **Method 3: Interactive Python**

```bash
# Activate environment
source venv/bin/activate

# Start Python
python

# Then run code interactively
>>> import pandas as pd
>>> import numpy as np
>>> from prophet import Prophet
>>> # ... your code here
```

## 📊 Testing and Validation

### **1. Run Installation Test**

```bash
source venv/bin/activate
python test_installation_simple.py
```

**Expected Output:**
```
✅ Pandas 2.3.3
✅ NumPy 1.26.4
✅ Matplotlib 3.9.4
✅ Seaborn 0.13.2
✅ Plotly 6.3.1
✅ Prophet
✅ Statsmodels ARIMA
✅ Scikit-learn
⚠️ XGBoost not available (optional)
```

### **2. Test Data Generation**

```bash
source venv/bin/activate
python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate test data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
costs = np.random.uniform(100, 1000, len(dates))

df = pd.DataFrame({
    'UsageDateTime': dates,
    'PreTaxCost': costs,
    'MeterCategory': np.random.choice(['Compute', 'Storage', 'Database'], len(dates))
})

print(f'Generated {len(df)} records')
print(f'Total cost: ${df[\"PreTaxCost\"].sum():,.2f}')
print('✅ Data generation working!')
"
```

### **3. Test Prophet Model**

```bash
source venv/bin/activate
python -c "
from prophet import Prophet
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = 100 + np.random.normal(0, 10, 100)

df = pd.DataFrame({'ds': dates, 'y': values})

# Train model
model = Prophet()
model.fit(df)

# Make forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

print(f'✅ Prophet working! Forecast: {len(forecast)} points')
"
```

## 🔍 Expected Results

### **Data Generation (Notebook 1)**
- ✅ 50,000 sample Azure cost records
- ✅ All required attributes generated
- ✅ Realistic cost patterns and seasonality
- ✅ Files saved to `data/` directory

### **Data Exploration (Notebook 2)**
- ✅ Cost analysis by category
- ✅ Time series patterns
- ✅ Seasonal analysis
- ✅ Data quality assessment
- ✅ Interactive visualizations

### **Prophet Model (Notebook 3)**
- ✅ Models trained for key categories
- ✅ 90-day future forecasts
- ✅ Confidence intervals
- ✅ Model evaluation metrics
- ✅ Results saved to `results/prophet/`

### **ARIMA Model (Notebook 4)**
- ✅ Stationarity tests
- ✅ Automatic parameter selection
- ✅ 30-day forecasts
- ✅ Model diagnostics
- ✅ Results saved to `results/arima/`

### **XGBoost Model (Notebook 5) - Optional**
- ✅ Feature engineering pipeline
- ✅ Time series cross-validation
- ✅ Feature importance analysis
- ✅ Results saved to `results/xgboost/`

### **Model Comparison (Notebook 6)**
- ✅ Performance comparison
- ✅ Forecast visualization
- ✅ Model recommendations
- ✅ Interactive dashboard

## 🚨 Troubleshooting

### **Common Issues:**

1. **XGBoost Error:**
   ```
   XGBoost Library (libxgboost.dylib) could not be loaded
   ```
   **Solution:** Install OpenMP: `brew install libomp`

2. **Prophet Installation Issues:**
   ```
   ImportError: No module named 'prophet'
   ```
   **Solution:** `pip install prophet`

3. **Memory Issues:**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution:** Reduce dataset size in notebooks

4. **Jupyter Not Starting:**
   ```
   command not found: jupyter
   ```
   **Solution:** `pip install jupyter`

### **Performance Tips:**

- **Start Small:** Use smaller datasets for testing
- **Close Plots:** Use `plt.close()` to free memory
- **Batch Processing:** Process data in chunks
- **Monitor Memory:** Use `htop` or Activity Monitor

## 📁 Project Structure

```
/Users/sabbineni/projects/acm/
├── notebooks/           # Jupyter notebooks
│   ├── 01_data_generation.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_prophet_model.ipynb
│   ├── 04_arima_model.ipynb
│   ├── 05_xgboost_model.ipynb
│   └── 06_model_comparison.ipynb
├── data/               # Generated data files
├── results/            # Model results
│   ├── prophet/
│   ├── arima/
│   └── xgboost/
├── utils/              # Utility functions
├── venv/               # Virtual environment
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

## 🎯 Quick Start Commands

```bash
# 1. Setup
cd /Users/sabbineni/projects/acm
source venv/bin/activate

# 2. Test installation
python test_installation_simple.py

# 3. Start Jupyter
jupyter lab

# 4. Open first notebook
# Navigate to: notebooks/01_data_generation.ipynb
```

## 📞 Support

If you encounter issues:

1. **Check the test script output**
2. **Verify virtual environment activation**
3. **Check Python version compatibility**
4. **Review error messages carefully**
5. **Try running notebooks cell by cell**

## 🎉 Success Indicators

You'll know everything is working when:

- ✅ All test scripts pass
- ✅ Jupyter starts without errors
- ✅ Data generation creates realistic Azure cost data
- ✅ Prophet and ARIMA models train successfully
- ✅ Visualizations display correctly
- ✅ Results are saved to appropriate directories

**Happy Forecasting! 🚀📊**


