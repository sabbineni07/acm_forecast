"""
Model Comparison Module
Section 6.1: Model Performance - Comparison
Section 5.2.2: Model Selection Criteria
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

from .performance_metrics import PerformanceMetrics
from .model_evaluator import ModelEvaluator
from ..config.settings import performance_config

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare multiple forecasting models
    Section 6.1: Model Performance
    Section 5.2.2: Model Selection Criteria
    """
    
    def __init__(self):
        """Initialize model comparator"""
        self.evaluator = ModelEvaluator()
        self.metrics_calculator = PerformanceMetrics()
    
    def compare_models(self,
                      forecasts: Dict[str, Union[pd.Series, pd.DataFrame, np.ndarray]],
                      actual: Union[pd.Series, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple models (Section 6.1)
        
        Args:
            forecasts: Dictionary of {model_name: forecast_values}
            actual: Actual values
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_results = {}
        
        for model_name, forecast in forecasts.items():
            metrics = self.evaluator.evaluate_model(forecast, actual, model_name)
            comparison_results[model_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        # Add ranking
        comparison_df['mape_rank'] = comparison_df['mape'].rank()
        comparison_df['r2_rank'] = comparison_df['r2'].rank(ascending=False)
        comparison_df['overall_rank'] = (
            comparison_df['mape_rank'] + comparison_df['r2_rank']
        ) / 2
        
        logger.info(f"Compared {len(forecasts)} models")
        return comparison_df
    
    def select_best_model(self,
                         comparison_df: pd.DataFrame,
                         metric: str = 'mape') -> str:
        """
        Select best model based on metric (Section 5.2.2)
        
        Args:
            comparison_df: Comparison DataFrame from compare_models()
            metric: Metric to use for selection ('mape', 'rmse', 'mae', 'r2')
            
        Returns:
            Name of best model
        """
        if metric == 'r2':
            best_model = comparison_df[metric].idxmax()
        else:
            best_model = comparison_df[metric].idxmin()
        
        logger.info(f"Selected best model: {best_model} based on {metric}")
        return best_model
    
    def create_ensemble_forecast(self,
                                forecasts: Dict[str, np.ndarray],
                                weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create ensemble forecast (Section 5.2.2)
        
        Args:
            forecasts: Dictionary of {model_name: forecast_values}
            weights: Optional weights for each model (default: equal weights)
            
        Returns:
            Ensemble forecast array
        """
        if weights is None:
            # Equal weights
            weights = {name: 1.0 / len(forecasts) for name in forecasts.keys()}
        
        # Ensure all forecasts have same length
        min_len = min(len(f) for f in forecasts.values())
        aligned_forecasts = {name: f[:min_len] for name, f in forecasts.items()}
        
        # Weighted average
        ensemble = np.zeros(min_len)
        total_weight = sum(weights.values())
        
        for name, forecast in aligned_forecasts.items():
            weight = weights.get(name, 0) / total_weight
            ensemble += forecast * weight
        
        logger.info(f"Created ensemble forecast from {len(forecasts)} models")
        return ensemble

