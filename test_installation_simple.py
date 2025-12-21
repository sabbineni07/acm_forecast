#!/usr/bin/env python3
"""
Simple test script to validate the Azure Cost Management forecasting project installation.
This script tests the core libraries and creates a simple data generation test.
"""

import sys
import traceback

def test_imports():
    """Test all required library imports."""
    print("🔍 Testing library imports...")
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {plt.matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from prophet import Prophet
        print("✅ Prophet")
    except ImportError as e:
        print(f"❌ Prophet import failed: {e}")
        return False
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        print("✅ Statsmodels ARIMA")
    except ImportError as e:
        print(f"❌ Statsmodels import failed: {e}")
        return False
    
    try:
        from sklearn.model_selection import TimeSeriesSplit
        print("✅ Scikit-learn")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    # Test XGBoost (optional)
    try:
        import xgboost as xgb
        print(f"✅ XGBoost {xgb.__version__}")
        return True
    except Exception as e:
        print(f"⚠️ XGBoost not available: {e}")
        print("   (This is optional - you can install OpenMP with: brew install libomp)")
        return True  # Don't fail the test for XGBoost
    
    return True

def test_data_generation():
    """Test basic data generation functionality."""
    print("\n🔍 Testing data generation...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import random
        
        # Generate simple test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        costs = np.random.uniform(100, 1000, len(dates))
        
        df = pd.DataFrame({
            'UsageDateTime': dates,
            'PreTaxCost': costs,
            'MeterCategory': np.random.choice(['Compute', 'Storage', 'Database'], len(dates))
        })
        
        print(f"✅ Generated test data: {len(df)} records")
        print(f"   Date range: {df['UsageDateTime'].min()} to {df['UsageDateTime'].max()}")
        print(f"   Total cost: ${df['PreTaxCost'].sum():,.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        traceback.print_exc()
        return None

def test_prophet_basic():
    """Test basic Prophet functionality."""
    print("\n🔍 Testing Prophet basic functionality...")
    
    try:
        from prophet import Prophet
        import pandas as pd
        import numpy as np
        
        # Create simple test data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        values = 100 + np.random.normal(0, 10, 100)
        
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        # Initialize and fit model
        model = Prophet()
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        print(f"✅ Prophet test successful")
        print(f"   Training data: {len(df)} points")
        print(f"   Forecast data: {len(forecast)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Prophet test failed: {e}")
        traceback.print_exc()
        return False

def test_arima_basic():
    """Test basic ARIMA functionality."""
    print("\n🔍 Testing ARIMA basic functionality...")
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import numpy as np
        
        # Create simple test data
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)
        
        # Fit ARIMA model
        model = ARIMA(data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=10)
        
        print(f"✅ ARIMA test successful")
        print(f"   Training data: {len(data)} points")
        print(f"   Forecast: {len(forecast)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ ARIMA test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization():
    """Test basic visualization functionality."""
    print("\n🔍 Testing visualization libraries...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import numpy as np
        
        # Test matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title('Test Plot')
        plt.close(fig)  # Close to avoid display issues
        
        # Test plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
        fig.update_layout(title='Test Plotly Plot')
        
        print("✅ Visualization libraries working")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Azure Cost Management Forecasting - Installation Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        sys.exit(1)
    
    # Test data generation
    test_data = test_data_generation()
    if test_data is None:
        print("\n❌ Data generation test failed.")
        sys.exit(1)
    
    # Test Prophet
    if not test_prophet_basic():
        print("\n⚠️ Prophet test failed, but continuing...")
    
    # Test ARIMA
    if not test_arima_basic():
        print("\n⚠️ ARIMA test failed, but continuing...")
    
    # Test visualization
    if not test_visualization():
        print("\n⚠️ Visualization test failed, but continuing...")
    
    print("\n" + "=" * 60)
    print("🎉 Installation test completed!")
    print("\n📋 Next Steps:")
    print("1. Run Jupyter: jupyter lab")
    print("2. Open notebooks in the 'notebooks' directory")
    print("3. Start with '01_data_generation.ipynb'")
    print("\n💡 Tips:")
    print("- Make sure to activate your virtual environment first")
    print("- If you encounter issues, check the error messages above")
    print("- Some libraries may take time to load on first use")
    print("- For XGBoost: Install OpenMP with 'brew install libomp' (optional)")

if __name__ == "__main__":
    main()


