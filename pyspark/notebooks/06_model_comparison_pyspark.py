# Azure Cost Management Model Comparison - PySpark Version

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AzureCostModelComparison") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("PySpark session initialized successfully!")
print(f"Spark version: {spark.version}")

# Load all model results
print("Loading model results...")

results = {}
model_types = ['prophet', 'arima', 'xgboost']

for model_type in model_types:
    try:
        results_path = f"/Users/sabbineni/projects/acm/pyspark/results/{model_type}/evaluation_results.pkl"
        with open(results_path, 'rb') as f:
            results[model_type] = pickle.load(f)
        print(f"✅ Loaded {model_type} results")
    except FileNotFoundError:
        print(f"⚠️ {model_type} results not found")
        results[model_type] = {}

# Load forecasting data for comparison
with open('/Users/sabbineni/projects/acm/pyspark/data/forecasting_data.pkl', 'rb') as f:
    forecasting_data = pickle.load(f)

print(f"Loaded forecasting data for categories: {list(forecasting_data.keys())}")

# Load forecasts
forecasts = {}
for model_type in model_types:
    forecasts[model_type] = {}
    for category in ['Total', 'Compute', 'Storage', 'Database']:
        try:
            if model_type == 'prophet':
                forecast_path = f"/Users/sabbineni/projects/acm/pyspark/results/{model_type}/forecast_{category.lower()}.csv"
            elif model_type == 'arima':
                forecast_path = f"/Users/sabbineni/projects/acm/pyspark/results/{model_type}/arima_forecast_{category.lower()}.csv"
            else:  # xgboost
                forecast_path = f"/Users/sabbineni/projects/acm/pyspark/results/{model_type}/xgboost_forecast_{category.lower()}.csv"
            
            forecast_df = pd.read_csv(forecast_path)
            forecasts[model_type][category] = forecast_df
            print(f"✅ Loaded {model_type} forecast for {category}")
        except FileNotFoundError:
            print(f"⚠️ {model_type} forecast for {category} not found")

# Model comparison functions
def create_performance_comparison(results):
    """Create comprehensive performance comparison"""
    print("\n=== Model Performance Comparison ===")
    
    # Combine all performance metrics
    all_metrics = {}
    
    for model_type, model_results in results.items():
        if model_results:
            for category, metrics in model_results.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in ['feature_importance']:  # Skip feature importance
                            key = f"{model_type}_{category}_{metric_name}"
                            all_metrics[key] = metric_value
    
    # Create comparison DataFrame
    comparison_data = []
    for model_type, model_results in results.items():
        if model_results:
            for category, metrics in model_results.items():
                if isinstance(metrics, dict):
                    row = {
                        'Model': model_type.upper(),
                        'Category': category,
                        'RMSE': metrics.get('rmse', metrics.get('test_rmse', np.nan)),
                        'MAE': metrics.get('mae', metrics.get('test_mae', np.nan)),
                        'MAPE': metrics.get('mape', np.nan),
                        'R2': metrics.get('r_squared', metrics.get('test_r2', np.nan)),
                        'AIC': metrics.get('aic', np.nan),
                        'BIC': metrics.get('bic', np.nan)
                    }
                    comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        print("\nPerformance Comparison Summary:")
        print(comparison_df.round(4))
        
        # Save comparison
        comparison_path = "/Users/sabbineni/projects/acm/pyspark/results/model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✅ Model comparison saved: {comparison_path}")
    
    return comparison_df

def create_forecast_comparison(forecasts, category='Total'):
    """Create forecast comparison visualization"""
    print(f"\n=== Forecast Comparison for {category} ===")
    
    if category not in forecasting_data:
        print(f"⚠️ No data available for {category}")
        return None
    
    # Get actual data
    actual_data = forecasting_data[category]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{category} - All Models Forecast',
            f'{category} - Model Comparison (Last 30 Days)',
            f'{category} - Forecast Accuracy',
            f'{category} - Model Performance Metrics'
        )
    )
    
    colors = {'prophet': 'blue', 'arima': 'green', 'xgboost': 'red'}
    
    # 1. All models forecast
    # Actual data
    fig.add_trace(
        go.Scatter(
            x=actual_data.index,
            y=actual_data.values,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    # Model forecasts
    for model_type, model_forecasts in forecasts.items():
        if category in model_forecasts:
            forecast_df = model_forecasts[category]
            if 'date' in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['forecast'],
                        mode='lines',
                        name=f'{model_type.upper()} Forecast',
                        line=dict(color=colors.get(model_type, 'gray'))
                    ),
                    row=1, col=1
                )
    
    # 2. Model comparison (last 30 days)
    last_30_days = actual_data.tail(30)
    fig.add_trace(
        go.Scatter(
            x=last_30_days.index,
            y=last_30_days.values,
            mode='lines',
            name='Actual (Last 30)',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    for model_type, model_forecasts in forecasts.items():
        if category in model_forecasts:
            forecast_df = model_forecasts[category]
            if 'date' in forecast_df.columns:
                # Get first 30 days of forecast
                first_30_forecast = forecast_df.head(30)
                fig.add_trace(
                    go.Scatter(
                        x=first_30_forecast['date'],
                        y=first_30_forecast['forecast'],
                        mode='lines',
                        name=f'{model_type.upper()} (First 30)',
                        line=dict(color=colors.get(model_type, 'gray')),
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # 3. Forecast accuracy (if we have performance metrics)
    if results:
        accuracy_data = []
        for model_type, model_results in results.items():
            if model_results and category in model_results:
                metrics = model_results[category]
                if isinstance(metrics, dict):
                    accuracy_data.append({
                        'Model': model_type.upper(),
                        'RMSE': metrics.get('rmse', metrics.get('test_rmse', np.nan)),
                        'MAE': metrics.get('mae', metrics.get('test_mae', np.nan)),
                        'R2': metrics.get('r_squared', metrics.get('test_r2', np.nan))
                    })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            
            # RMSE comparison
            fig.add_trace(
                go.Bar(
                    x=accuracy_df['Model'],
                    y=accuracy_df['RMSE'],
                    name='RMSE',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
    
    # 4. Model performance metrics
    if results:
        performance_data = []
        for model_type, model_results in results.items():
            if model_results and category in model_results:
                metrics = model_results[category]
                if isinstance(metrics, dict):
                    performance_data.append({
                        'Model': model_type.upper(),
                        'AIC': metrics.get('aic', np.nan),
                        'BIC': metrics.get('bic', np.nan)
                    })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            
            # AIC comparison
            fig.add_trace(
                go.Bar(
                    x=performance_df['Model'],
                    y=performance_df['AIC'],
                    name='AIC',
                    marker_color='lightgreen'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=800,
        title_text=f"Model Comparison - {category}",
        showlegend=True
    )
    
    fig.show()
    return fig

def create_model_selection_guide(comparison_df):
    """Create model selection guide based on performance"""
    print("\n=== Model Selection Guide ===")
    
    if comparison_df.empty:
        print("⚠️ No comparison data available")
        return
    
    # Group by model and calculate average performance
    model_performance = comparison_df.groupby('Model').agg({
        'RMSE': 'mean',
        'MAE': 'mean',
        'R2': 'mean',
        'AIC': 'mean'
    }).round(4)
    
    print("\nAverage Performance by Model:")
    print(model_performance)
    
    # Model selection recommendations
    print("\n🎯 Model Selection Recommendations:")
    
    # Best RMSE
    best_rmse = model_performance['RMSE'].idxmin()
    print(f"📊 Best RMSE: {best_rmse} ({model_performance.loc[best_rmse, 'RMSE']:.2f})")
    
    # Best MAE
    best_mae = model_performance['MAE'].idxmin()
    print(f"📊 Best MAE: {best_mae} ({model_performance.loc[best_mae, 'MAE']:.2f})")
    
    # Best R2
    best_r2 = model_performance['R2'].idxmax()
    print(f"📊 Best R²: {best_r2} ({model_performance.loc[best_r2, 'R2']:.4f})")
    
    # Best AIC (lowest is best)
    best_aic = model_performance['AIC'].idxmin()
    print(f"📊 Best AIC: {best_aic} ({model_performance.loc[best_aic, 'AIC']:.2f})")
    
    # Overall recommendation
    print(f"\n🏆 Overall Recommendation:")
    print(f"   For Production: {best_rmse} (best overall accuracy)")
    print(f"   For Research: {best_r2} (best explanatory power)")
    print(f"   For Simplicity: ARIMA (statistical model)")
    print(f"   For Scalability: XGBOOST (handles complex patterns)")
    
    return model_performance

def create_interactive_dashboard(comparison_df, forecasts):
    """Create interactive dashboard for model comparison"""
    print("\n=== Creating Interactive Dashboard ===")
    
    # Create dashboard with multiple tabs
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Model Performance Overview',
            'Forecast Comparison - Total',
            'Forecast Comparison - Compute',
            'Forecast Comparison - Storage',
            'Forecast Comparison - Database',
            'Model Selection Guide'
        )
    )
    
    # 1. Model Performance Overview
    if not comparison_df.empty:
        # Average RMSE by model
        model_avg = comparison_df.groupby('Model')['RMSE'].mean()
        fig.add_trace(
            go.Bar(
                x=model_avg.index,
                y=model_avg.values,
                name='Average RMSE',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
    
    # 2-5. Forecast comparisons for each category
    categories = ['Total', 'Compute', 'Storage', 'Database']
    colors = {'prophet': 'blue', 'arima': 'green', 'xgboost': 'red'}
    
    for i, category in enumerate(categories):
        row = 1 if i < 2 else 2
        col = 2 if i < 2 else (i - 1)
        
        if category in forecasting_data:
            actual_data = forecasting_data[category]
            
            # Actual data
            fig.add_trace(
                go.Scatter(
                    x=actual_data.index,
                    y=actual_data.values,
                    mode='lines',
                    name=f'{category} Actual',
                    line=dict(color='black', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Model forecasts
            for model_type, model_forecasts in forecasts.items():
                if category in model_forecasts:
                    forecast_df = model_forecasts[category]
                    if 'date' in forecast_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_df['date'],
                                y=forecast_df['forecast'],
                                mode='lines',
                                name=f'{model_type.upper()}',
                                line=dict(color=colors.get(model_type, 'gray')),
                                showlegend=False
                            ),
                            row=row, col=col
                        )
    
    # 6. Model Selection Guide
    if not comparison_df.empty:
        model_performance = comparison_df.groupby('Model').agg({
            'RMSE': 'mean',
            'R2': 'mean'
        }).round(4)
        
        # Create radar chart data
        models = model_performance.index.tolist()
        rmse_values = model_performance['RMSE'].values
        r2_values = model_performance['R2'].values
        
        # Normalize values for radar chart
        rmse_norm = 1 - (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min())
        r2_norm = (r2_values - r2_values.min()) / (r2_values.max() - r2_values.min())
        
        for i, model in enumerate(models):
            fig.add_trace(
                go.Scatter(
                    x=['RMSE', 'R2'],
                    y=[rmse_norm[i], r2_norm[i]],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=3),
                    showlegend=False
                ),
                row=2, col=3
            )
    
    fig.update_layout(
        height=1000,
        title_text="Azure Cost Management - Model Comparison Dashboard",
        showlegend=True
    )
    
    fig.show()
    return fig

# Run comparisons
print("\n" + "="*80)
print("AZURE COST MANAGEMENT - MODEL COMPARISON ANALYSIS")
print("="*80)

# 1. Performance comparison
comparison_df = create_performance_comparison(results)

# 2. Forecast comparisons for each category
for category in ['Total', 'Compute', 'Storage', 'Database']:
    create_forecast_comparison(forecasts, category)

# 3. Model selection guide
model_performance = create_model_selection_guide(comparison_df)

# 4. Interactive dashboard
dashboard = create_interactive_dashboard(comparison_df, forecasts)

# Save final results
print("\n=== Saving Final Results ===")

import os
results_dir = "/Users/sabbineni/projects/acm/pyspark/results"
os.makedirs(results_dir, exist_ok=True)

# Save model performance summary
if model_performance is not None:
    performance_path = f"{results_dir}/final_model_performance.csv"
    model_performance.to_csv(performance_path)
    print(f"✅ Final model performance saved: {performance_path}")

# Save comprehensive comparison
if not comparison_df.empty:
    comprehensive_path = f"{results_dir}/comprehensive_model_comparison.csv"
    comparison_df.to_csv(comprehensive_path, index=False)
    print(f"✅ Comprehensive comparison saved: {comprehensive_path}")

# Create summary report
summary_report = {
    'total_models_trained': len([m for m in results.values() if m]),
    'categories_analyzed': list(forecasting_data.keys()),
    'best_overall_model': comparison_df.groupby('Model')['RMSE'].mean().idxmin() if not comparison_df.empty else 'N/A',
    'average_rmse': comparison_df['RMSE'].mean() if not comparison_df.empty else 'N/A',
    'average_r2': comparison_df['R2'].mean() if not comparison_df.empty else 'N/A'
}

summary_path = f"{results_dir}/model_comparison_summary.pkl"
with open(summary_path, 'wb') as f:
    pickle.dump(summary_report, f)
print(f"✅ Summary report saved: {summary_path}")

print("\n🎉 Model comparison analysis completed successfully!")
print("📊 All results saved to pyspark/results/")
print("🔮 Comprehensive model evaluation completed")
print("🎯 Model selection recommendations provided")

# Display final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"Total models trained: {summary_report['total_models_trained']}")
print(f"Categories analyzed: {len(summary_report['categories_analyzed'])}")
print(f"Best overall model: {summary_report['best_overall_model']}")
print(f"Average RMSE: {summary_report['average_rmse']:.2f}")
print(f"Average R²: {summary_report['average_r2']:.4f}")

spark.stop()


