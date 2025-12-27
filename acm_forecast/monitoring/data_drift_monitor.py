"""
Data Drift Monitoring Module
Section 8.2: Data Drift Monitoring
"""

from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """
    Monitor data drift
    Section 8.2: Data Drift Monitoring
    """
    
    def __init__(self):
        """Initialize data drift monitor"""
        self.baseline_distributions = {}
    
    def set_baseline(self,
                    data: pd.DataFrame,
                    column: str,
                    name: str = "baseline") -> None:
        """
        Set baseline distribution for drift detection
        
        Args:
            data: Baseline data
            column: Column to monitor
            name: Name for this baseline
        """
        if column not in data.columns:
            raise ValueError(f"Column {column} not found in data")
        
        self.baseline_distributions[name] = {
            "column": column,
            "data": data[column],
            "mean": data[column].mean(),
            "std": data[column].std(),
            "distribution": data[column].values
        }
        
        logger.info(f"Set baseline for {name} ({column})")
    
    def detect_distribution_drift(self,
                                 current_data: pd.DataFrame,
                                 baseline_name: str = "baseline",
                                 method: str = "ks") -> Dict[str, Any]:
        """
        Detect distribution drift (Section 8.2)
        
        Args:
            current_data: Current data to compare
            baseline_name: Name of baseline to compare against
            method: Detection method ('ks' for Kolmogorov-Smirnov, 'chi2' for Chi-square)
            
        Returns:
            Dictionary with drift detection results
        """
        if baseline_name not in self.baseline_distributions:
            raise ValueError(f"Baseline {baseline_name} not found")
        
        baseline = self.baseline_distributions[baseline_name]
        column = baseline["column"]
        
        if column not in current_data.columns:
            raise ValueError(f"Column {column} not found in current data")
        
        current_values = current_data[column].dropna()
        baseline_values = baseline["data"].dropna()
        
        if method == "ks":
            # Kolmogorov-Smirnov test
            statistic, pvalue = stats.ks_2samp(baseline_values, current_values)
            is_drift = pvalue < 0.05
        elif method == "chi2":
            # Chi-square test (for categorical)
            # This would need binning for continuous variables
            is_drift = False  # Placeholder
            statistic = None
            pvalue = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result = {
            "baseline_name": baseline_name,
            "column": column,
            "method": method,
            "statistic": statistic,
            "pvalue": pvalue,
            "is_drift": is_drift,
            "drift_magnitude": abs(current_values.mean() - baseline["mean"]) / baseline["std"] if baseline["std"] > 0 else 0,
            "timestamp": datetime.now()
        }
        
        logger.info(
            f"Drift detection for {column}: "
            f"Drift={is_drift}, p-value={pvalue:.4f}"
        )
        
        return result
    
    def detect_feature_drift(self,
                           current_data: pd.DataFrame,
                           features: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift across multiple features
        
        Args:
            current_data: Current data
            features: List of features to check (default: all baselines)
            
        Returns:
            Dictionary of drift results by feature
        """
        if features is None:
            features = list(self.baseline_distributions.keys())
        
        results = {}
        for baseline_name in features:
            try:
                drift_result = self.detect_distribution_drift(
                    current_data, baseline_name
                )
                results[baseline_name] = drift_result
            except Exception as e:
                logger.warning(f"Could not detect drift for {baseline_name}: {e}", exc_info=True)
        
        return results


