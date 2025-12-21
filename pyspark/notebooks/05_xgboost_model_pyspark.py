# Azure Cost Management XGBoost Forecasting - PySpark Version

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AzureCostXGBoostForecasting") \
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

# Feature engineering functions
def create_time_features(df):
    """Create time-based features from datetime index"""
    df_features = df.copy()
    
    # Extract time components
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['day'] = df_features.index.day
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['dayofyear'] = df_features.index.dayofyear
    df_features['quarter'] = df_features.index.quarter
    df_features['week'] = df_features.index.isocalendar().week
    
    # Cyclical encoding for time features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)
    df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
    df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
    
    # Weekend indicator
    df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)
    
    # Holiday indicators (simplified)
    df_features['is_holiday'] = 0  # Could be enhanced with actual holiday data
    
    return df_features

def create_lag_features(df, target_col, lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for time series"""
    df_lags = df.copy()
    
    for lag in lags:
        df_lags[f'{target_col}_lag_{lag}'] = df_lags[target_col].shift(lag)
    
    return df_lags

def create_rolling_features(df, target_col, windows=[7, 14, 30]):
    """Create rolling window features"""
    df_rolling = df.copy()
    
    for window in windows:
        df_rolling[f'{target_col}_rolling_mean_{window}'] = df_rolling[target_col].rolling(window=window).mean()
        df_rolling[f'{target_col}_rolling_std_{window}'] = df_rolling[target_col].rolling(window=window).std()
        df_rolling[f'{target_col}_rolling_min_{window}'] = df_rolling[target_col].rolling(window=window).min()
        df_rolling[f'{target_col}_rolling_max_{window}'] = df_rolling[target_col].rolling(window=window).max()
    
    return df_rolling

def create_expanding_features(df, target_col):
    """Create expanding window features"""
    df_expanding = df.copy()
    
    df_expanding[f'{target_col}_expanding_mean'] = df_expanding[target_col].expanding().mean()
    df_expanding[f'{target_col}_expanding_std'] = df_expanding[target_col].expanding().std()
    df_expanding[f'{target_col}_expanding_min'] = df_expanding[target_col].expanding().min()
    df_expanding[f'{target_col}_expanding_max'] = df_expanding[target_col].expanding().max()
    
    return df_expanding

def create_interaction_features(df, target_col):
    """Create interaction features"""
    df_interactions = df.copy()
    
    # Time-based interactions
    df_interactions['month_dayofweek'] = df_interactions['month'] * df_interactions['dayofweek']
    df_interactions['quarter_month'] = df_interactions['quarter'] * df_interactions['month']
    
    # Lag interactions
    if f'{target_col}_lag_1' in df_interactions.columns:
        df_interactions['lag1_rolling7'] = df_interactions[f'{target_col}_lag_1'] * df_interactions[f'{target_col}_rolling_mean_7']
        df_interactions['lag7_rolling30'] = df_interactions[f'{target_col}_lag_7'] * df_interactions[f'{target_col}_rolling_mean_30']
    
    return df_interactions

def prepare_features(timeseries, target_col='value'):
    """Prepare comprehensive feature set for XGBoost"""
    print(f"Preparing features for {target_col}...")
    
    # Convert to DataFrame
    df = pd.DataFrame({target_col: timeseries})
    
    # Create all feature types
    df = create_time_features(df)
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = create_expanding_features(df, target_col)
    df = create_interaction_features(df, target_col)
    
    # Remove rows with NaN values (from lag and rolling features)
    df = df.dropna()
    
    print(f"✅ Features created: {df.shape[1]} features, {df.shape[0]} samples")
    return df

def train_xgboost_model(features_df, target_col, category_name, test_size=0.2):
    """Train XGBoost model with time series cross-validation"""
    print(f"\n--- Training XGBoost model for {category_name} ---")
    
    # Prepare features and target
    feature_cols = [col for col in features_df.columns if col != target_col]
    X = features_df[feature_cols]
    y = features_df[target_col]
    
    # Time series split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"✅ XGBoost model trained successfully")
    print(f"   Training RMSE: {train_rmse:.2f}")
    print(f"   Test RMSE: {test_rmse:.2f}")
    print(f"   Training MAE: {train_mae:.2f}")
    print(f"   Test MAE: {test_mae:.2f}")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    
    return model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }

def generate_forecasts(model, features_df, target_col, forecast_steps=90):
    """Generate future forecasts using trained XGBoost model"""
    print(f"Generating {forecast_steps} day forecast...")
    
    # Get the last known values
    last_row = features_df.iloc[-1].copy()
    last_date = features_df.index[-1]
    
    forecasts = []
    forecast_dates = []
    
    # Generate forecasts step by step
    for i in range(forecast_steps):
        # Create next date
        next_date = last_date + pd.Timedelta(days=i+1)
        forecast_dates.append(next_date)
        
        # Update time features for next date
        next_row = last_row.copy()
        next_row['year'] = next_date.year
        next_row['month'] = next_date.month
        next_row['day'] = next_date.day
        next_row['dayofweek'] = next_date.dayofweek
        next_row['dayofyear'] = next_date.dayofyear
        next_row['quarter'] = next_date.quarter
        next_row['week'] = next_date.isocalendar().week
        
        # Update cyclical features
        next_row['month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
        next_row['month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
        next_row['day_sin'] = np.sin(2 * np.pi * next_date.day / 31)
        next_row['day_cos'] = np.cos(2 * np.pi * next_date.day / 31)
        next_row['dayofweek_sin'] = np.sin(2 * np.pi * next_date.dayofweek / 7)
        next_row['dayofweek_cos'] = np.cos(2 * np.pi * next_date.dayofweek / 7)
        next_row['quarter_sin'] = np.sin(2 * np.pi * next_date.quarter / 4)
        next_row['quarter_cos'] = np.cos(2 * np.pi * next_date.quarter / 4)
        
        # Update weekend indicator
        next_row['is_weekend'] = 1 if next_date.dayofweek >= 5 else 0
        
        # Update interaction features
        next_row['month_dayofweek'] = next_date.month * next_date.dayofweek
        next_row['quarter_month'] = next_date.quarter * next_date.month
        
        # Make prediction
        feature_cols = [col for col in features_df.columns if col != target_col]
        X_next = next_row[feature_cols].values.reshape(1, -1)
        forecast_value = model.predict(X_next)[0]
        forecasts.append(forecast_value)
        
        # Update lag features for next iteration
        last_row[target_col] = forecast_value
        for lag in [1, 2, 3, 7, 14, 30]:
            if f'{target_col}_lag_{lag}' in last_row.index:
                if lag == 1:
                    last_row[f'{target_col}_lag_{lag}'] = forecast_value
                else:
                    # Shift other lags
                    if f'{target_col}_lag_{lag-1}' in last_row.index:
                        last_row[f'{target_col}_lag_{lag}'] = last_row[f'{target_col}_lag_{lag-1}']
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecasts
    })
    
    print(f"✅ Forecast generated for {forecast_steps} days")
    print(f"   Forecast range: {forecast_dates[0]} to {forecast_dates[-1]}")
    print(f"   Average forecast: ${np.mean(forecasts):.2f}")
    
    return forecast_df

def plot_xgboost_analysis(features_df, model, forecast_df, category_name):
    """Create comprehensive XGBoost analysis visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'{category_name} - Actual vs Predicted',
            f'{category_name} - Forecast',
            f'{category_name} - Feature Importance (Top 10)',
            f'{category_name} - Residuals',
            f'{category_name} - Residuals Distribution',
            f'{category_name} - Prediction vs Actual Scatter'
        )
    )
    
    # Get predictions for training data
    feature_cols = [col for col in features_df.columns if col != 'value']
    X = features_df[feature_cols]
    y_pred = model.predict(X)
    y_actual = features_df['value']
    
    # 1. Actual vs Predicted
    fig.add_trace(
        go.Scatter(
            x=features_df.index, 
            y=y_actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=features_df.index, 
            y=y_pred,
            mode='lines',
            name='Predicted',
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
    
    # 3. Feature Importance
    feature_importance = model.feature_importances_
    feature_names = feature_cols
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(10)
    
    fig.add_trace(
        go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            name='Feature Importance',
            marker_color='purple'
        ),
        row=2, col=1
    )
    
    # 4. Residuals
    residuals = y_actual - y_pred
    fig.add_trace(
        go.Scatter(
            x=features_df.index, 
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='orange')
        ),
        row=2, col=2
    )
    
    # 5. Residuals Distribution
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name='Residuals Distribution',
            nbinsx=30,
            marker_color='lightblue'
        ),
        row=3, col=1
    )
    
    # 6. Prediction vs Actual Scatter
    fig.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_pred,
            mode='markers',
            name='Pred vs Actual',
            marker=dict(color='red', size=4, opacity=0.6)
        ),
        row=3, col=2
    )
    
    # Add diagonal line for perfect prediction
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        height=1200,
        title_text=f"XGBoost Model Analysis - {category_name}",
        showlegend=True
    )
    
    fig.show()

# Train models for key categories
key_categories = ['Total', 'Compute', 'Storage', 'Database']
xgboost_models = {}
xgboost_forecasts = {}
xgboost_evaluations = {}

print("\n=== Training XGBoost Models ===")

for category in key_categories:
    if category in forecasting_data:
        print(f"\n{'='*60}")
        print(f"PROCESSING CATEGORY: {category}")
        print(f"{'='*60}")
        
        timeseries = forecasting_data[category]
        
        if len(timeseries) > 100:  # Need sufficient data for feature engineering
            # Prepare features
            features_df = prepare_features(timeseries, 'value')
            
            if len(features_df) > 50:  # Need sufficient data after feature engineering
                # Train XGBoost model
                model, evaluation = train_xgboost_model(features_df, 'value', category)
                
                # Generate forecasts
                forecast_df = generate_forecasts(model, features_df, 'value')
                
                # Store results
                xgboost_models[category] = model
                xgboost_forecasts[category] = forecast_df
                xgboost_evaluations[category] = evaluation
                
                # Create visualization
                plot_xgboost_analysis(features_df, model, forecast_df, category)
                
            else:
                print(f"⚠️ Insufficient data after feature engineering for {category}: {len(features_df)} points")
        else:
            print(f"⚠️ Insufficient data for {category}: {len(timeseries)} points")
    else:
        print(f"⚠️ No data available for {category}")

# Save results
print("\n=== Saving Results ===")

import os
results_dir = "/Users/sabbineni/projects/acm/pyspark/results/xgboost"
os.makedirs(results_dir, exist_ok=True)

# Save models
for category, model in xgboost_models.items():
    model_path = f"{results_dir}/xgboost_model_{category.lower()}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ XGBoost model saved: {model_path}")

# Save forecasts
for category, forecast in xgboost_forecasts.items():
    forecast_path = f"{results_dir}/xgboost_forecast_{category.lower()}.csv"
    forecast.to_csv(forecast_path, index=False)
    print(f"✅ XGBoost forecast saved: {forecast_path}")

# Save evaluation results
evaluation_path = f"{results_dir}/xgboost_evaluation_results.pkl"
with open(evaluation_path, 'wb') as f:
    pickle.dump(xgboost_evaluations, f)
print(f"✅ Evaluation results saved: {evaluation_path}")

# Create performance comparison
print("\n=== Model Performance Summary ===")

if xgboost_evaluations:
    # Create performance DataFrame
    performance_data = {}
    for category, eval_results in xgboost_evaluations.items():
        performance_data[category] = {
            'Train_RMSE': eval_results['train_rmse'],
            'Test_RMSE': eval_results['test_rmse'],
            'Train_MAE': eval_results['train_mae'],
            'Test_MAE': eval_results['test_mae'],
            'Train_R2': eval_results['train_r2'],
            'Test_R2': eval_results['test_r2']
        }
    
    performance_df = pd.DataFrame(performance_data).T
    performance_df = performance_df.round(4)
    print("\nXGBoost Model Performance:")
    print(performance_df)
    
    # Save performance summary
    performance_path = f"{results_dir}/xgboost_performance_summary.csv"
    performance_df.to_csv(performance_path)
    print(f"✅ Performance summary saved: {performance_path}")

# Create feature importance comparison
print("\n=== Feature Importance Analysis ===")

if xgboost_evaluations:
    # Combine feature importance from all models
    all_features = set()
    for eval_results in xgboost_evaluations.values():
        all_features.update(eval_results['feature_importance'].keys())
    
    importance_df = pd.DataFrame(index=sorted(all_features))
    for category, eval_results in xgboost_evaluations.items():
        importance_df[category] = importance_df.index.map(eval_results['feature_importance']).fillna(0)
    
    # Get top 20 most important features across all models
    importance_df['avg_importance'] = importance_df.mean(axis=1)
    top_features = importance_df.nlargest(20, 'avg_importance')
    
    print("\nTop 20 Most Important Features:")
    print(top_features[['avg_importance']].round(4))
    
    # Save feature importance
    importance_path = f"{results_dir}/feature_importance_summary.csv"
    importance_df.to_csv(importance_path)
    print(f"✅ Feature importance saved: {importance_path}")

print("\n🎉 XGBoost model training completed successfully!")
print("📊 All results saved to pyspark/results/xgboost/")
print("🔮 Future forecasts generated for 90 days ahead")
print("🎯 Feature importance analysis completed")

spark.stop()


