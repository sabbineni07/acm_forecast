"""
Performance Monitoring Module
Section 8.1: Performance Monitoring
"""

from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..config import AppConfig
from ..evaluation.performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor model performance in production
    Section 8.1: Performance Monitoring
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize performance monitor
        
        Args:
            config: AppConfig instance containing configuration
        """
        self.config = config
        self.metrics_calculator = PerformanceMetrics()
        self.performance_history = []
    
    def check_forecast_accuracy(self,
                               forecast: np.ndarray,
                               actual: np.ndarray,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Check forecast accuracy (Section 8.1)
        
        Args:
            forecast: Forecast values
            actual: Actual values
            model_name: Name of the model
            
        Returns:
            Dictionary with accuracy check results
        """
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(actual, forecast)
        
        # Check against thresholds
        critical_mape = self.config.performance.critical_mape or 15.0
        warning_mape = self.config.performance.warning_mape or 12.0
        target_mape = self.config.performance.target_mape or 10.0
        
        alert_level = "OK"
        if metrics['mape'] > critical_mape:
            alert_level = "CRITICAL"
        elif metrics['mape'] > warning_mape:
            alert_level = "WARNING"
        
        result = {
            "model_name": model_name,
            "timestamp": datetime.now(),
            "metrics": metrics,
            "alert_level": alert_level,
            "meets_target": metrics['mape'] < target_mape
        }
        
        # Store in history
        self.performance_history.append(result)
        
        logger.info(
            f"Performance check for {model_name}: "
            f"MAPE={metrics['mape']:.2f}%, Alert={alert_level}"
        )
        
        return result
    
    def track_performance_trends(self, days: int = 30) -> pd.DataFrame:
        """
        Track performance trends over time (Section 8.1)
        
        Args:
            days: Number of days to analyze
            
        Returns:
            DataFrame with performance trends
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_history)
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]
        
        # Calculate trends
        if len(df) > 0:
            df['mape_trend'] = df['metrics'].apply(lambda x: x['mape']).rolling(7).mean()
            df['r2_trend'] = df['metrics'].apply(lambda x: x['r2']).rolling(7).mean()
        
        return df
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report (Section 8.4)
        
        Returns:
            Dictionary with performance report
        """
        if not self.performance_history:
            return {"status": "no_data", "message": "No performance data available"}
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate summary statistics
        mape_values = df['metrics'].apply(lambda x: x['mape'])
        r2_values = df['metrics'].apply(lambda x: x['r2'])
        
        report = {
            "report_date": datetime.now(),
            "total_checks": len(df),
            "average_mape": mape_values.mean(),
            "average_r2": r2_values.mean(),
            "min_mape": mape_values.min(),
            "max_mape": mape_values.max(),
            "alerts": {
                "critical": len(df[df['alert_level'] == 'CRITICAL']),
                "warning": len(df[df['alert_level'] == 'WARNING']),
                "ok": len(df[df['alert_level'] == 'OK'])
            },
            "meets_target_percentage": (df['meets_target'].sum() / len(df) * 100) if len(df) > 0 else 0
        }
        
        logger.info(f"Generated performance report: {report}")
        return report


