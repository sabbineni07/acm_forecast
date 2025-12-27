"""
ARIMA Model Plugin

PRIMARY IMPLEMENTATION for ARIMA time series forecasting.
The actual implementation is here - ARIMAForecaster class delegates to this.
"""

from typing import Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
import logging

from ...core.interfaces import IModel
from ...core.base_plugin import BasePlugin
from ...config import AppConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ARIMAModelPlugin(BasePlugin, IModel):
    """
    ARIMA model plugin - PRIMARY IMPLEMENTATION
    Section 4.2.1: ARIMA Model Methodology
    Section 5.2.1: Model Estimation - ARIMA
    Section 5.2.3: Final Model Specification - ARIMA
    Section 5.2.4: Model Diagnostics - ARIMA
    """
    
    def __init__(self, config: AppConfig, category: str = "Total", **kwargs):
        """Initialize ARIMA model plugin
        
        Args:
            config: AppConfig instance
            category: Cost category name
            **kwargs: Plugin-specific configuration
        """
        super().__init__(config, None, **kwargs)  # ARIMA doesn't need Spark
        self.category = category
        self.model = None
        self.is_trained = False
        self.order = None
        self.seasonal_order = None
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ARIMA model (Section 5.2.1)
        
        Args:
            df: Training DataFrame or Series (time series)
            
        Returns:
            Dictionary with training results
        """
        # Convert DataFrame to Series if needed (for ARIMA, we expect Series)
        if isinstance(df, pd.DataFrame):
            # Assume first column is the time series
            timeseries = df.iloc[:, 0] if len(df.columns) > 0 else df.squeeze()
        else:
            timeseries = df
        
        logger.info(f"Training ARIMA model for {self.category} with {len(timeseries)} records")
        
        # Auto-select ARIMA parameters
        arima_config = self.config.model.arima
        self.model = auto_arima(
            timeseries,
            seasonal=arima_config.seasonal or True,
            m=arima_config.seasonal_period or 12,
            max_p=arima_config.max_p or 5,
            max_d=arima_config.max_d or 2,
            max_q=arima_config.max_q or 5,
            information_criterion=arima_config.information_criterion or "aic",
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
        self.is_trained = True
        
        logger.info(
            f"ARIMA model trained for {self.category}: "
            f"order={self.order}, seasonal_order={self.seasonal_order}"
        )
        
        return {
            "model": self.model,
            "is_trained": True,
            "category": self.category,
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }
    
    def predict(self, periods: int = 30, **kwargs) -> pd.DataFrame:
        """
        Generate predictions (Section 5.2.1)
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional parameters (return_conf_int, etc.)
            
        Returns:
            DataFrame or Series with forecasts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return_conf_int = kwargs.get('return_conf_int', True)
        
        # Generate forecast
        forecast, conf_int = self.model.predict(
            n_periods=periods,
            return_conf_int=return_conf_int
        )
        
        logger.info(f"Generated {periods} period forecast for {self.category}")
        
        # Convert to DataFrame
        if return_conf_int and conf_int is not None:
            result_df = pd.DataFrame({
                'forecast': forecast,
                'lower': conf_int[:, 0],
                'upper': conf_int[:, 1]
            })
        else:
            result_df = pd.DataFrame({'forecast': forecast})
        
        return result_df
    
    def save(self, path: str) -> None:
        """
        Save model to path
        
        Args:
            path: Path to save model
        """
        if self.model is not None:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved ARIMA model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from path
        
        Args:
            path: Path to load model from
        """
        import pickle
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Loaded ARIMA model from {path}")
    
    # Additional methods for backward compatibility
    def test_stationarity(self, timeseries: pd.Series) -> Dict[str, Any]:
        """Test stationarity (Section 5.2.4)"""
        # ADF test
        adf_result = adfuller(timeseries.dropna())
        
        # KPSS test
        try:
            kpss_result = kpss(timeseries.dropna())
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
        except Exception as e:
            logger.warning(f"KPSS test failed for {self.category}: {e}")
            kpss_stat = None
            kpss_pvalue = None
        
        results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_is_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_stat,
            'kpss_pvalue': kpss_pvalue,
            'kpss_is_stationary': kpss_pvalue > 0.05 if kpss_pvalue else None
        }
        
        logger.info(f"Stationarity test for {self.category}: {results}")
        return results
    
    def make_stationary(self, 
                        timeseries: pd.Series,
                        max_diff: int = 3) -> Tuple[pd.Series, int]:
        """Make time series stationary via differencing"""
        diff_order = 0
        stationary_series = timeseries.copy()
        
        for i in range(max_diff):
            adf_result = adfuller(stationary_series.dropna())
            if adf_result[1] < 0.05:  # Stationary
                break
            stationary_series = stationary_series.diff()
            diff_order += 1
        
        logger.info(f"Applied {diff_order} order differencing for {self.category}")
        return stationary_series, diff_order
    
    def forecast(self, n_periods: int = 30, return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """Alias for predict() for backward compatibility"""
        result = self.predict(periods=n_periods, return_conf_int=return_conf_int)
        if return_conf_int and 'lower' in result.columns:
            return result['forecast'], result[['lower', 'upper']]
        return result['forecast'], None
    
    def evaluate(self, forecast: pd.Series, actual: pd.Series) -> Dict[str, float]:
        """Evaluate model on test data"""
        # Calculate metrics
        y_true = actual.values
        y_pred = forecast.values if isinstance(forecast, pd.Series) else forecast
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
        logger.info(f"Evaluation metrics for {self.category}: {metrics}")
        return metrics
