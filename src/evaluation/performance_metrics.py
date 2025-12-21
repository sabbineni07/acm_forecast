"""
Performance Metrics Module
Section 6.1: Model Performance
"""

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate performance metrics for forecasting models
    Section 6.1: Model Performance
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive forecast metrics (Section 5.2.2)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        # Remove any NaN or inf values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'mae_percentage': np.nan
            }
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        r2 = r2_score(y_true_clean, y_pred_clean)
        mae_percentage = (mae / np.mean(y_true_clean)) * 100 if np.mean(y_true_clean) > 0 else np.nan
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'mae_percentage': mae_percentage
        }
        
        return metrics
    
    @staticmethod
    def calculate_by_horizon(y_true: pd.Series,
                            y_pred: pd.Series,
                            horizons: list = [1, 7, 30, 90]) -> Dict[int, Dict[str, float]]:
        """
        Calculate metrics by forecast horizon (Section 6.1)
        
        Args:
            y_true: Actual values with datetime index
            y_pred: Predicted values with datetime index
            horizons: List of forecast horizons (days)
            
        Returns:
            Dictionary of metrics by horizon
        """
        results = {}
        
        for horizon in horizons:
            if len(y_true) >= horizon:
                # Get first 'horizon' predictions
                y_true_h = y_true.iloc[:horizon]
                y_pred_h = y_pred.iloc[:horizon]
                
                metrics = PerformanceMetrics.calculate_metrics(
                    y_true_h.values, y_pred_h.values
                )
                results[horizon] = metrics
        
        return results
    
    @staticmethod
    def create_performance_summary(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create performance summary table (Section 6.1)
        
        Args:
            metrics_dict: Dictionary of {model_name: {metric: value}}
            
        Returns:
            DataFrame with performance summary
        """
        df = pd.DataFrame(metrics_dict).T
        
        # Round numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df


