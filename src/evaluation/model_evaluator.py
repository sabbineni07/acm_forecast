"""
Model Evaluator Module
Section 6.1: Model Performance
Section 6.2: Sensitivity Analyses
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

from .performance_metrics import PerformanceMetrics
from ..config.settings import performance_config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance
    Section 6.1: Model Performance
    """
    
    def __init__(self):
        """Initialize model evaluator"""
        self.metrics_calculator = PerformanceMetrics()
    
    def evaluate_model(self,
                      forecast: Union[pd.Series, pd.DataFrame, np.ndarray],
                      actual: Union[pd.Series, pd.DataFrame, np.ndarray],
                      model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model performance (Section 6.1)
        
        Args:
            forecast: Forecast values
            actual: Actual values
            model_name: Name of the model
            
        Returns:
            Dictionary of performance metrics
        """
        # Convert to numpy arrays
        if isinstance(forecast, pd.DataFrame):
            forecast = forecast.iloc[:, 0].values if len(forecast.columns) > 0 else forecast.values
        elif isinstance(forecast, pd.Series):
            forecast = forecast.values
        
        if isinstance(actual, pd.DataFrame):
            actual = actual.iloc[:, 0].values if len(actual.columns) > 0 else actual.values
        elif isinstance(actual, pd.Series):
            actual = actual.values
        
        # Align lengths
        min_len = min(len(forecast), len(actual))
        forecast = forecast[:min_len]
        actual = actual[:min_len]
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(actual, forecast)
        
        # Add performance assessment
        metrics['meets_target_mape'] = metrics['mape'] < performance_config.target_mape
        metrics['meets_target_r2'] = metrics['r2'] > performance_config.target_r2
        
        logger.info(f"Evaluated {model_name}: MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
        return metrics
    
    def sensitivity_analysis(self,
                           base_forecast: np.ndarray,
                           base_actual: np.ndarray,
                           perturbations: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis (Section 6.2)
        
        Args:
            base_forecast: Base forecast values
            base_actual: Actual values
            perturbations: Dictionary of {perturbation_name: perturbed_forecast}
            
        Returns:
            Dictionary of metrics for each perturbation
        """
        # Base metrics
        base_metrics = self.metrics_calculator.calculate_metrics(base_actual, base_forecast)
        
        results = {'base': base_metrics}
        
        # Perturbed metrics
        for name, perturbed_forecast in perturbations.items():
            metrics = self.metrics_calculator.calculate_metrics(base_actual, perturbed_forecast)
            results[name] = metrics
        
        logger.info(f"Sensitivity analysis completed for {len(perturbations)} perturbations")
        return results
    
    def benchmark_comparison(self,
                           model_forecast: np.ndarray,
                           actual: np.ndarray,
                           naive_forecasts: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Compare model with naive benchmarks (Section 6.3)
        
        Args:
            model_forecast: Model forecast values
            actual: Actual values
            naive_forecasts: Dictionary of {method_name: forecast_values}
            
        Returns:
            Dictionary of metrics for model and benchmarks
        """
        # Model metrics
        model_metrics = self.metrics_calculator.calculate_metrics(actual, model_forecast)
        results = {'model': model_metrics}
        
        # Benchmark metrics
        for name, naive_forecast in naive_forecasts.items():
            metrics = self.metrics_calculator.calculate_metrics(actual, naive_forecast)
            results[name] = metrics
        
        logger.info(f"Benchmark comparison completed with {len(naive_forecasts)} benchmarks")
        return results

