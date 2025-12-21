# Azure Cost Management Prophet Forecasting - PySpark Version

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AzureCostProphetForecasting") \
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

# Prophet model functions
def prepare_prophet_data(ts_data, category_name):
    """Prepare time series data for Prophet model"""
    # Convert to DataFrame with required columns
    df_prophet = pd.DataFrame({
        'ds': ts_data.index,
        'y': ts_data.values
    })
    
    # Remove any non-positive values
    df_prophet = df_prophet[df_prophet['y'] > 0]
    
    print(f"{category_name}: {len(df_prophet)} data points for Prophet")
    return df_prophet

def train_prophet_model(df_prophet, category_name):
    """Train Prophet model and generate forecasts"""
    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    # Add custom seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add US holidays
    model.add_country_holidays(country_name='US')
    
    # Fit the model
    model.fit(df_prophet)
    
    # Create future dataframe for 90 days
    future = model.make_future_dataframe(periods=90, freq='D')
    
    # Generate forecast
    forecast = model.predict(future)
    
    print(f"✅ Prophet model trained for {category_name}")
    print(f"   Training data: {len(df_prophet)} points")
    print(f"   Forecast data: {len(forecast)} points")
    
    return model, forecast

def evaluate_prophet_model(model, df_prophet, category_name):
    """Evaluate Prophet model using cross-validation"""
    try:
        # Perform cross-validation
        df_cv = cross_validation(
            model, 
            initial='180 days', 
            period='30 days', 
            horizon='30 days',
            parallel="threads"
        )
        
        # Calculate performance metrics
        df_performance = performance_metrics(df_cv)
        
        # Extract key metrics
        rmse = df_performance['rmse'].mean()
        mae = df_performance['mae'].mean()
        mape = df_performance['mape'].mean()
        coverage = df_performance['coverage'].mean()
        
        print(f"✅ Model evaluation for {category_name}:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Coverage: {coverage:.2f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'coverage': coverage
        }
        
    except Exception as e:
        print(f"⚠️ Cross-validation failed for {category_name}: {e}")
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'coverage': np.nan
        }

def plot_prophet_forecast(model, forecast, df_prophet, category_name):
    """Create Prophet forecast visualization"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{category_name} - Forecast vs Actual',
            f'{category_name} - Trend',
            f'{category_name} - Weekly Seasonality',
            f'{category_name} - Yearly Seasonality'
        )
    )
    
    # 1. Forecast vs Actual
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_lower'],
            mode='lines',
            name='Lower Bound',
            line=dict(color='lightblue', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='lightblue', dash='dash'),
            fill='tonexty',
            fillcolor='rgba(173,216,230,0.3)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_prophet['ds'], 
            y=df_prophet['y'],
            mode='markers',
            name='Actual',
            marker=dict(color='red', size=4)
        ),
        row=1, col=1
    )
    
    # 2. Trend
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='green')
        ),
        row=1, col=2
    )
    
    # 3. Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['weekly'],
                mode='lines',
                name='Weekly',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
    
    # 4. Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yearly'],
                mode='lines',
                name='Yearly',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=f"Prophet Model Analysis - {category_name}",
        showlegend=True
    )
    
    fig.show()

# Train models for key categories
key_categories = ['Total', 'Compute', 'Storage', 'Database']
models = {}
forecasts = {}
evaluation_results = {}

print("\n=== Training Prophet Models ===")

for category in key_categories:
    if category in forecasting_data:
        print(f"\n--- Training model for {category} ---")
        
        # Prepare data
        df_prophet = prepare_prophet_data(forecasting_data[category], category)
        
        if len(df_prophet) > 30:  # Need sufficient data for training
            # Train model
            model, forecast = train_prophet_model(df_prophet, category)
            
            # Evaluate model
            evaluation = evaluate_prophet_model(model, df_prophet, category)
            
            # Store results
            models[category] = model
            forecasts[category] = forecast
            evaluation_results[category] = evaluation
            
            # Create visualization
            plot_prophet_forecast(model, forecast, df_prophet, category)
            
        else:
            print(f"⚠️ Insufficient data for {category}: {len(df_prophet)} points")
    else:
        print(f"⚠️ No data available for {category}")

# Generate future forecasts
print("\n=== Generating Future Forecasts ===")

future_forecasts = {}
for category in key_categories:
    if category in models:
        model = models[category]
        
        # Create future dataframe for next 90 days
        future = model.make_future_dataframe(periods=90, freq='D')
        
        # Generate forecast
        future_forecast = model.predict(future)
        
        # Extract only future predictions
        future_only = future_forecast[future_forecast['ds'] > forecasting_data[category].index.max()]
        
        future_forecasts[category] = future_only
        
        print(f"✅ Future forecast generated for {category}: {len(future_only)} days")
        print(f"   Forecast range: {future_only['ds'].min()} to {future_only['ds'].max()}")
        print(f"   Average predicted cost: ${future_only['yhat'].mean():.2f}")

# Save results
print("\n=== Saving Results ===")

import os
results_dir = "/Users/sabbineni/projects/acm/pyspark/results/prophet"
os.makedirs(results_dir, exist_ok=True)

# Save models
for category, model in models.items():
    model_path = f"{results_dir}/prophet_model_{category.lower()}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path}")

# Save forecasts
for category, forecast in forecasts.items():
    forecast_path = f"{results_dir}/forecast_{category.lower()}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"✅ Forecast saved: {forecast_path}")

# Save future forecasts
for category, future_forecast in future_forecasts.items():
    future_path = f"{results_dir}/future_forecast_{category.lower()}.csv"
    future_forecast.to_csv(future_path, index=False)
    print(f"✅ Future forecast saved: {future_path}")

# Save evaluation results
evaluation_path = f"{results_dir}/evaluation_results.pkl"
with open(evaluation_path, 'wb') as f:
    pickle.dump(evaluation_results, f)
print(f"✅ Evaluation results saved: {evaluation_path}")

# Create summary comparison
print("\n=== Model Performance Summary ===")
evaluation_df = pd.DataFrame(evaluation_results).T
evaluation_df = evaluation_df.round(4)
print(evaluation_df)

# Save performance summary
performance_path = f"{results_dir}/performance_summary.csv"
evaluation_df.to_csv(performance_path)
print(f"✅ Performance summary saved: {performance_path}")

# Create forecast comparison visualization
print("\n=== Creating Forecast Comparison ===")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Total Cost Forecast', 'Compute Cost Forecast', 
                   'Storage Cost Forecast', 'Database Cost Forecast')
)

colors = ['blue', 'green', 'orange', 'purple']
for i, category in enumerate(key_categories):
    if category in forecasts:
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        forecast = forecasts[category]
        
        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'],
                mode='lines',
                name=f'{category} Forecast',
                line=dict(color=colors[i])
            ),
            row=row, col=col
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_upper'],
                mode='lines',
                name=f'{category} Upper',
                line=dict(color=colors[i], dash='dash'),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_lower'],
                mode='lines',
                name=f'{category} Lower',
                line=dict(color=colors[i], dash='dash'),
                fill='tonexty',
                fillcolor=f'rgba({colors[i]},0.3)',
                showlegend=False
            ),
            row=row, col=col
        )

fig.update_layout(
    height=800,
    title_text="Prophet Model Forecasts Comparison",
    showlegend=True
)

fig.show()

print("\n🎉 Prophet model training and evaluation completed successfully!")
print("📊 All results saved to pyspark/results/prophet/")
print("🔮 Future forecasts generated for 90 days ahead")

spark.stop()
