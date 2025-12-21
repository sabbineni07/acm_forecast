"""
Utility functions for Azure Cost Management forecasting project.

This module contains helper functions for data processing, model evaluation,
and visualization that are used across multiple notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive forecast evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary containing various metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'MAE_Percentage': (mean_absolute_error(y_true, y_pred) / np.mean(y_true)) * 100
    }
    
    return metrics


def create_time_series_features(df: pd.DataFrame, date_col: str = 'UsageDateTime') -> pd.DataFrame:
    """
    Create comprehensive time-based features for time series forecasting.
    
    Args:
        df: DataFrame with datetime column
        date_col: Name of the datetime column
    
    Returns:
        DataFrame with additional time-based features
    """
    df_features = df.copy()
    
    # Basic time features
    df_features['Year'] = df_features[date_col].dt.year
    df_features['Month'] = df_features[date_col].dt.month
    df_features['Day'] = df_features[date_col].dt.day
    df_features['DayOfWeek'] = df_features[date_col].dt.dayofweek
    df_features['DayOfYear'] = df_features[date_col].dt.dayofyear
    df_features['WeekOfYear'] = df_features[date_col].dt.isocalendar().week
    df_features['Quarter'] = df_features[date_col].dt.quarter
    
    # Boolean features
    df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
    df_features['IsMonthStart'] = df_features[date_col].dt.is_month_start.astype(int)
    df_features['IsMonthEnd'] = df_features[date_col].dt.is_month_end.astype(int)
    df_features['IsQuarterStart'] = df_features[date_col].dt.is_quarter_start.astype(int)
    df_features['IsQuarterEnd'] = df_features[date_col].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding
    df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
    df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
    df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
    df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
    df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
    
    return df_features


def create_lag_features(df: pd.DataFrame, target_col: str, group_col: str, 
                       lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lag features for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        group_col: Name of the grouping column (e.g., 'MeterCategory')
        lags: List of lag periods to create
    
    Returns:
        DataFrame with lag features
    """
    df_lags = df.copy()
    df_lags = df_lags.sort_values('UsageDateTime')
    
    for lag in lags:
        df_lags[f'{target_col}_lag_{lag}'] = df_lags.groupby(group_col)[target_col].shift(lag)
    
    return df_lags


def create_rolling_features(df: pd.DataFrame, target_col: str, group_col: str,
                          windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create rolling window features for time series forecasting.
    
    Args:
        df: DataFrame with time series data
        target_col: Name of the target column
        group_col: Name of the grouping column
        windows: List of window sizes
    
    Returns:
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    df_rolling = df_rolling.sort_values('UsageDateTime')
    
    for window in windows:
        df_rolling[f'{target_col}_rolling_mean_{window}'] = (
            df_rolling.groupby(group_col)[target_col]
            .rolling(window=window)
            .mean()
            .reset_index(0, drop=True)
        )
        df_rolling[f'{target_col}_rolling_std_{window}'] = (
            df_rolling.groupby(group_col)[target_col]
            .rolling(window=window)
            .std()
            .reset_index(0, drop=True)
        )
        df_rolling[f'{target_col}_rolling_max_{window}'] = (
            df_rolling.groupby(group_col)[target_col]
            .rolling(window=window)
            .max()
            .reset_index(0, drop=True)
        )
        df_rolling[f'{target_col}_rolling_min_{window}'] = (
            df_rolling.groupby(group_col)[target_col]
            .rolling(window=window)
            .min()
            .reset_index(0, drop=True)
        )
    
    return df_rolling


def plot_forecast_comparison(historical_data: pd.DataFrame, 
                           forecasts: Dict[str, pd.DataFrame],
                           title: str = "Forecast Comparison") -> None:
    """
    Create a comprehensive forecast comparison plot.
    
    Args:
        historical_data: DataFrame with historical data
        forecasts: Dictionary of forecast DataFrames
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(historical_data['UsageDateTime'], historical_data['PreTaxCost'], 
             label='Historical', color='black', linewidth=2)
    
    # Plot forecasts
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
        if 'ds' in forecast_df.columns:
            date_col = 'ds'
        elif 'date' in forecast_df.columns:
            date_col = 'date'
        else:
            date_col = forecast_df.columns[0]  # Use first column as date
        
        if 'yhat' in forecast_df.columns:
            pred_col = 'yhat'
        elif 'forecast' in forecast_df.columns:
            pred_col = 'forecast'
        else:
            pred_col = forecast_df.columns[1]  # Use second column as prediction
        
        plt.plot(forecast_df[date_col], forecast_df[pred_col], 
                label=f'{model_name} Forecast', color=colors[i % len(colors)], 
                linewidth=2, linestyle='--')
        
        # Add confidence intervals if available
        if 'yhat_upper' in forecast_df.columns and 'yhat_lower' in forecast_df.columns:
            plt.fill_between(forecast_df[date_col], 
                           forecast_df['yhat_lower'], 
                           forecast_df['yhat_upper'],
                           alpha=0.2, color=colors[i % len(colors)])
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cost ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def create_model_performance_table(performance_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a formatted performance comparison table.
    
    Args:
        performance_data: Dictionary with model performance metrics
    
    Returns:
        Formatted DataFrame with performance metrics
    """
    df = pd.DataFrame(performance_data).T
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    return df


def save_model_results(model, model_name: str, category: str, 
                      results_dir: str = '/Users/sabbineni/projects/acm/results') -> None:
    """
    Save model results to organized directory structure.
    
    Args:
        model: Trained model object
        model_name: Name of the model (e.g., 'prophet', 'arima', 'xgboost')
        category: Category name (e.g., 'total', 'compute')
        results_dir: Base results directory
    """
    import joblib
    import os
    
    # Create directory structure
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{model_name}_model_{category}.pkl")
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")


def load_model_results(model_name: str, category: str,
                      results_dir: str = '/Users/sabbineni/projects/acm/results'):
    """
    Load saved model results.
    
    Args:
        model_name: Name of the model
        category: Category name
        results_dir: Base results directory
    
    Returns:
        Loaded model object
    """
    import joblib
    
    model_path = os.path.join(results_dir, model_name, f"{model_name}_model_{category}.pkl")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at: {model_path}")


def validate_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Perform comprehensive data quality validation.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_records': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'date_range': {
            'start': df['UsageDateTime'].min() if 'UsageDateTime' in df.columns else None,
            'end': df['UsageDateTime'].max() if 'UsageDateTime' in df.columns else None
        }
    }
    
    # Check for outliers in cost data
    if 'PreTaxCost' in df.columns:
        Q1 = df['PreTaxCost'].quantile(0.25)
        Q3 = df['PreTaxCost'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['PreTaxCost'] < Q1 - 1.5 * IQR) | 
                     (df['PreTaxCost'] > Q3 + 1.5 * IQR)]
        validation_results['outliers'] = len(outliers)
        validation_results['outlier_percentage'] = len(outliers) / len(df) * 100
    
    return validation_results


def print_validation_summary(validation_results: Dict[str, any]) -> None:
    """
    Print a formatted validation summary.
    
    Args:
        validation_results: Results from validate_data_quality function
    """
    print("=== DATA QUALITY VALIDATION SUMMARY ===")
    print(f"Total Records: {validation_results['total_records']:,}")
    print(f"Duplicate Records: {validation_results['duplicate_records']}")
    print(f"Memory Usage: {validation_results['memory_usage']:.2f} MB")
    
    if validation_results['date_range']['start']:
        print(f"Date Range: {validation_results['date_range']['start']} to {validation_results['date_range']['end']}")
    
    if 'outliers' in validation_results:
        print(f"Outliers: {validation_results['outliers']} ({validation_results['outlier_percentage']:.2f}%)")
    
    print("\nMissing Values:")
    for col, missing_count in validation_results['missing_values'].items():
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/validation_results['total_records']*100:.2f}%)")
    
    print("\nData Types:")
    for col, dtype in validation_results['data_types'].items():
        print(f"  {col}: {dtype}")


# Example usage and testing
if __name__ == "__main__":
    print("Azure Cost Management Data Utils")
    print("This module provides utility functions for the forecasting project.")
    print("Import this module in your notebooks to use these functions.")
