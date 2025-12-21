# Azure Cost Management ARIMA Forecasting - PySpark Version

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AzureCostARIMAForecasting") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("PySpark session initialized successfully!")
print(f"Spark version: {spark.version}")

# Load forecasting data
print("Loading forecasting data...")
with open('/Users/sabbineni/projects/acm/pyspark/data/forecasting_data.pkl', 'rb') as f:
    forecasting_data = pickle.load(f)

print(f"Loaded forecasting data for categories: {list(forecasting_data.keys())}")

# ARIMA model functions
def test_stationarity(timeseries, title="Time Series"):
    """Test stationarity using ADF and KPSS tests"""
    print(f"\n=== Stationarity Tests for {title} ===")
    
    # ADF Test
    adf_result = adfuller(timeseries.dropna())
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    if adf_result[1] <= 0.05:
        print("✅ ADF Test: Series is stationary (p-value <= 0.05)")
    else:
        print("❌ ADF Test: Series is non-stationary (p-value > 0.05)")
    
    # KPSS Test
    try:
        kpss_result = kpss(timeseries.dropna(), regression='c')
        print(f"\nKPSS Statistic: {kpss_result[0]:.6f}")
        print(f"p-value: {kpss_result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"\t{key}: {value:.3f}")
        
        if kpss_result[1] >= 0.05:
            print("✅ KPSS Test: Series is stationary (p-value >= 0.05)")
        else:
            print("❌ KPSS Test: Series is non-stationary (p-value < 0.05)")
    except Exception as e:
        print(f"⚠️ KPSS Test failed: {e}")
    
    return adf_result, kpss_result if 'kpss_result' in locals() else None

def make_stationary(timeseries, max_diff=3):
    """Make time series stationary through differencing"""
    print(f"\n=== Making Series Stationary ===")
    
    original_series = timeseries.copy()
    diff_count = 0
    
    for i in range(max_diff):
        adf_result = adfuller(timeseries.dropna())
        print(f"Differencing {i}: ADF p-value = {adf_result[1]:.6f}")
        
        if adf_result[1] <= 0.05:
            print(f"✅ Series is stationary after {i} differencing(s)")
            break
        
        timeseries = timeseries.diff().dropna()
        diff_count = i + 1
    
    if diff_count == max_diff:
        print(f"⚠️ Maximum differencing ({max_diff}) reached")
    
    return timeseries, diff_count

def find_arima_order(timeseries, max_p=5, max_d=2, max_q=5):
    """Find optimal ARIMA order using AIC"""
    print(f"\n=== Finding Optimal ARIMA Order ===")
    
    best_aic = float('inf')
    best_order = None
    results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    results.append((p, d, q, aic))
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                    
                    print(f"ARIMA({p},{d},{q}) - AIC: {aic:.2f}")
                    
                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) - Error: {e}")
                    continue
    
    print(f"\n✅ Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    return best_order, results

def find_sarima_order(timeseries, seasonal_period=12, max_p=2, max_d=1, max_q=2, max_P=1, max_D=1, max_Q=1):
    """Find optimal SARIMA order using AIC"""
    print(f"\n=== Finding Optimal SARIMA Order ===")
    
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(timeseries, 
                                               order=(p, d, q),
                                               seasonal_order=(P, D, Q, seasonal_period))
                                fitted_model = model.fit(disp=False)
                                aic = fitted_model.aic
                                results.append(((p, d, q), (P, D, Q, seasonal_period), aic))
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, seasonal_period)
                                
                                print(f"SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period}) - AIC: {aic:.2f}")
                                
                            except Exception as e:
                                print(f"SARIMA({p},{d},{q})({P},{D},{Q},{seasonal_period}) - Error: {e}")
                                continue
    
    print(f"\n✅ Best SARIMA order: {best_order} x {best_seasonal_order} with AIC: {best_aic:.2f}")
    return best_order, best_seasonal_order, results

def train_arima_model(timeseries, order, category_name):
    """Train ARIMA model and generate forecasts"""
    print(f"\n--- Training ARIMA model for {category_name} ---")
    
    try:
        # Fit ARIMA model
        model = ARIMA(timeseries, order=order)
        fitted_model = model.fit()
        
        print(f"✅ ARIMA model trained successfully")
        print(f"   Order: {order}")
        print(f"   AIC: {fitted_model.aic:.2f}")
        print(f"   BIC: {fitted_model.bic:.2f}")
        print(f"   Log-Likelihood: {fitted_model.llf:.2f}")
        
        # Generate forecasts
        forecast_steps = 90
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Create forecast dates
        last_date = timeseries.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        })
        
        print(f"✅ Forecast generated for {forecast_steps} days")
        print(f"   Forecast range: {forecast_dates[0]} to {forecast_dates[-1]}")
        print(f"   Average forecast: ${forecast.mean():.2f}")
        
        return fitted_model, forecast_df
        
    except Exception as e:
        print(f"❌ ARIMA model training failed: {e}")
        return None, None

def train_sarima_model(timeseries, order, seasonal_order, category_name):
    """Train SARIMA model and generate forecasts"""
    print(f"\n--- Training SARIMA model for {category_name} ---")
    
    try:
        # Fit SARIMA model
        model = SARIMAX(timeseries, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        print(f"✅ SARIMA model trained successfully")
        print(f"   Order: {order}")
        print(f"   Seasonal Order: {seasonal_order}")
        print(f"   AIC: {fitted_model.aic:.2f}")
        print(f"   BIC: {fitted_model.bic:.2f}")
        print(f"   Log-Likelihood: {fitted_model.llf:.2f}")
        
        # Generate forecasts
        forecast_steps = 90
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        # Create forecast dates
        last_date = timeseries.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1]
        })
        
        print(f"✅ Forecast generated for {forecast_steps} days")
        print(f"   Forecast range: {forecast_dates[0]} to {forecast_dates[-1]}")
        print(f"   Average forecast: ${forecast.mean():.2f}")
        
        return fitted_model, forecast_df
        
    except Exception as e:
        print(f"❌ SARIMA model training failed: {e}")
        return None, None

def evaluate_arima_model(model, timeseries, category_name):
    """Evaluate ARIMA model performance"""
    print(f"\n--- Evaluating ARIMA model for {category_name} ---")
    
    try:
        # Get fitted values
        fitted_values = model.fittedvalues
        
        # Calculate residuals
        residuals = timeseries - fitted_values
        
        # Calculate performance metrics
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / timeseries)) * 100
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((timeseries - np.mean(timeseries))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"✅ Model evaluation completed:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²: {r_squared:.4f}")
        print(f"   AIC: {model.aic:.2f}")
        print(f"   BIC: {model.bic:.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r_squared': r_squared,
            'aic': model.aic,
            'bic': model.bic
        }
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return None

def plot_arima_analysis(timeseries, model, forecast_df, category_name):
    """Create comprehensive ARIMA analysis visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'{category_name} - Original vs Fitted',
            f'{category_name} - Forecast',
            f'{category_name} - Residuals',
            f'{category_name} - Residuals Distribution',
            f'{category_name} - ACF of Residuals',
            f'{category_name} - PACF of Residuals'
        )
    )
    
    # 1. Original vs Fitted
    fig.add_trace(
        go.Scatter(
            x=timeseries.index, 
            y=timeseries.values,
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=model.fittedvalues.index, 
            y=model.fittedvalues.values,
            mode='lines',
            name='Fitted',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # 2. Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='lightgreen', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='lightgreen', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(144,238,144,0.3)',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Residuals
    residuals = timeseries - model.fittedvalues
    fig.add_trace(
        go.Scatter(
            x=residuals.index, 
            y=residuals.values,
            mode='lines',
            name='Residuals',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    # 4. Residuals Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals.values,
            name='Residuals Distribution',
            nbinsx=30
        ),
        row=2, col=2
    )
    
    # 5. ACF of Residuals
    from statsmodels.tsa.stattools import acf
    acf_values = acf(residuals.dropna(), nlags=20)
    fig.add_trace(
        go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            name='ACF',
            marker_color='purple'
        ),
        row=3, col=1
    )
    
    # 6. PACF of Residuals
    from statsmodels.tsa.stattools import pacf
    pacf_values = pacf(residuals.dropna(), nlags=20)
    fig.add_trace(
        go.Bar(
            x=list(range(len(pacf_values))),
            y=pacf_values,
            name='PACF',
            marker_color='brown'
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        height=1200,
        title_text=f"ARIMA Model Analysis - {category_name}",
        showlegend=True
    )
    
    fig.show()

# Train models for key categories
key_categories = ['Total', 'Compute', 'Storage', 'Database']
arima_models = {}
sarima_models = {}
arima_forecasts = {}
sarima_forecasts = {}
arima_evaluations = {}
sarima_evaluations = {}

print("\n=== Training ARIMA and SARIMA Models ===")

for category in key_categories:
    if category in forecasting_data:
        print(f"\n{'='*60}")
        print(f"PROCESSING CATEGORY: {category}")
        print(f"{'='*60}")
        
        timeseries = forecasting_data[category]
        
        if len(timeseries) > 30:  # Need sufficient data
            # Test stationarity
            adf_result, kpss_result = test_stationarity(timeseries, category)
            
            # Make stationary if needed
            stationary_series, diff_count = make_stationary(timeseries.copy())
            
            # Find optimal ARIMA order
            arima_order, arima_results = find_arima_order(stationary_series)
            
            # Find optimal SARIMA order
            sarima_order, sarima_seasonal_order, sarima_results = find_sarima_order(stationary_series)
            
            # Train ARIMA model
            arima_model, arima_forecast = train_arima_model(timeseries, arima_order, f"{category} ARIMA")
            
            if arima_model is not None:
                arima_eval = evaluate_arima_model(arima_model, timeseries, f"{category} ARIMA")
                arima_models[category] = arima_model
                arima_forecasts[category] = arima_forecast
                arima_evaluations[category] = arima_eval
                
                # Create visualization
                plot_arima_analysis(timeseries, arima_model, arima_forecast, f"{category} ARIMA")
            
            # Train SARIMA model
            sarima_model, sarima_forecast = train_sarima_model(timeseries, sarima_order, sarima_seasonal_order, f"{category} SARIMA")
            
            if sarima_model is not None:
                sarima_eval = evaluate_arima_model(sarima_model, timeseries, f"{category} SARIMA")
                sarima_models[category] = sarima_model
                sarima_forecasts[category] = sarima_forecast
                sarima_evaluations[category] = sarima_eval
                
                # Create visualization
                plot_arima_analysis(timeseries, sarima_model, sarima_forecast, f"{category} SARIMA")
            
        else:
            print(f"⚠️ Insufficient data for {category}: {len(timeseries)} points")
    else:
        print(f"⚠️ No data available for {category}")

# Save results
print("\n=== Saving Results ===")

import os
results_dir = "/Users/sabbineni/projects/acm/pyspark/results/arima"
os.makedirs(results_dir, exist_ok=True)

# Save ARIMA models
for category, model in arima_models.items():
    model_path = f"{results_dir}/arima_model_{category.lower()}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ ARIMA model saved: {model_path}")

# Save SARIMA models
for category, model in sarima_models.items():
    model_path = f"{results_dir}/sarima_model_{category.lower()}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ SARIMA model saved: {model_path}")

# Save forecasts
for category, forecast in arima_forecasts.items():
    forecast_path = f"{results_dir}/arima_forecast_{category.lower()}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"✅ ARIMA forecast saved: {forecast_path}")

for category, forecast in sarima_forecasts.items():
    forecast_path = f"{results_dir}/sarima_forecast_{category.lower()}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"✅ SARIMA forecast saved: {forecast_path}")

# Save evaluation results
evaluation_path = f"{results_dir}/arima_evaluation_results.pkl"
with open(evaluation_path, 'wb') as f:
    pickle.dump({
        'arima_evaluations': arima_evaluations,
        'sarima_evaluations': sarima_evaluations
    }, f)
print(f"✅ Evaluation results saved: {evaluation_path}")

# Create performance comparison
print("\n=== Model Performance Summary ===")

if arima_evaluations:
    arima_df = pd.DataFrame(arima_evaluations).T
    arima_df = arima_df.round(4)
    print("\nARIMA Model Performance:")
    print(arima_df)

if sarima_evaluations:
    sarima_df = pd.DataFrame(sarima_evaluations).T
    sarima_df = sarima_df.round(4)
    print("\nSARIMA Model Performance:")
    print(sarima_df)

# Save performance summaries
if arima_evaluations:
    arima_performance_path = f"{results_dir}/arima_performance_summary.csv"
    arima_df.to_csv(arima_performance_path)
    print(f"✅ ARIMA performance summary saved: {arima_performance_path}")

if sarima_evaluations:
    sarima_performance_path = f"{results_dir}/sarima_performance_summary.csv"
    sarima_df.to_csv(sarima_performance_path)
    print(f"✅ SARIMA performance summary saved: {sarima_performance_path}")

print("\n🎉 ARIMA and SARIMA model training completed successfully!")
print("📊 All results saved to pyspark/results/arima/")
print("🔮 Future forecasts generated for 90 days ahead")

spark.stop()


