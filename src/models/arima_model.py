"""
ARIMA Model Implementation
Section 4.2.1: Model Methodology - ARIMA
Section 5.2.1: Model Estimation - ARIMA
Section 5.2.3: Final Model Specification - ARIMA
Section 5.2.4: Model Diagnostics - ARIMA
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

from ..config import AppConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA model for time series forecasting
    Section 4.2.1: ARIMA Model Methodology
    """
    
    def __init__(self, config: AppConfig, category: str = "Total"):
        """
        Initialize ARIMA forecaster
        
        Args:
            config: AppConfig instance containing configuration
            category: Cost category name
        """
        self.config = config
        self.category = category
        self.model = None
        self.is_trained = False
        self.order = None
        self.seasonal_order = None
    
    def test_stationarity(self, timeseries: pd.Series) -> Dict[str, Any]:
        """
        Test stationarity (Section 5.2.4)
        
        Args:
            timeseries: Time series data
            
        Returns:
            Dictionary with test results
        """
        # ADF test
        adf_result = adfuller(timeseries.dropna())
        
        # KPSS test
        try:
            kpss_result = kpss(timeseries.dropna())
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
        except:
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
        """
        Make time series stationary via differencing
        
        Args:
            timeseries: Time series data
            max_diff: Maximum differencing order
            
        Returns:
            Tuple of (stationary series, differencing order)
        """
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
    
    def train(self, timeseries: pd.Series) -> Any:
        """
        Train ARIMA model (Section 5.2.1)
        
        Args:
            timeseries: Time series data
            
        Returns:
            Trained ARIMA model
        """
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
        return self.model
    
    def predict(self, 
                n_periods: int = 30,
                return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate forecasts (Section 5.2.1)
        
        Args:
            n_periods: Number of periods to forecast
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Tuple of (forecast series, confidence intervals DataFrame)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Generate forecast
        forecast = self.model.predict(n_periods=n_periods, return_conf_int=return_conf_int)
        
        if return_conf_int:
            forecast_values = forecast[0]
            conf_int = forecast[1]
            conf_df = pd.DataFrame({
                'lower': conf_int[:, 0],
                'upper': conf_int[:, 1]
            })
        else:
            forecast_values = forecast
            conf_df = None
        
        logger.info(f"Generated {n_periods} period forecast for {self.category}")
        return forecast_values, conf_df
    
    def diagnose_residuals(self, timeseries: pd.Series) -> Dict[str, Any]:
        """
        Diagnose model residuals (Section 5.2.4)
        
        Args:
            timeseries: Time series data
            
        Returns:
            Dictionary with diagnostic results
        """
        if not self.is_trained:
            self.train(timeseries)
        
        # Get residuals
        residuals = self.model.resid()
        
        # Ljung-Box test for autocorrelation
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_pvalue = stats.shapiro(residuals.dropna())
        
        diagnostics = {
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'ljung_box_statistic': ljung_box['lb_stat'].iloc[-1],
            'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
            'ljung_box_is_white_noise': ljung_box['lb_pvalue'].iloc[-1] > 0.05,
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_pvalue,
            'shapiro_is_normal': shapiro_pvalue > 0.05
        }
        
        logger.info(f"Residual diagnostics for {self.category}: {diagnostics}")
        return diagnostics
    
    def evaluate(self,
                 forecast: pd.Series,
                 actual: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            forecast: Forecast values
            actual: Actual values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Align indices
        aligned = pd.DataFrame({
            'forecast': forecast,
            'actual': actual
        }).dropna()
        
        y_true = aligned['actual'].values
        y_pred = aligned['forecast'].values
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        logger.info(f"Evaluation metrics for {self.category}: {metrics}")
        return metrics


