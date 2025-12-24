"""
Prophet Model Implementation
Section 4.2.1: Model Methodology - Prophet
Section 5.2.1: Model Estimation - Prophet
Section 5.2.3: Final Model Specification - Prophet
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging

from ..config import AppConfig

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Prophet model for time series forecasting
    Section 4.2.1: Prophet Model Methodology
    """
    
    def __init__(self, config: AppConfig, category: str = "Total"):
        """
        Initialize Prophet forecaster
        
        Args:
            config: AppConfig instance containing configuration
            category: Cost category name
        """
        self.config = config
        self.category = category
        self.model = None
        self.is_trained = False
        
    def create_model(self) -> Prophet:
        """
        Create Prophet model with configuration (Section 5.2.3)
        
        Returns:
            Configured Prophet model
        """
        prophet_config = self.config.model.prophet
        model = Prophet(
            yearly_seasonality=prophet_config.yearly_seasonality or True,
            weekly_seasonality=prophet_config.weekly_seasonality or True,
            daily_seasonality=prophet_config.daily_seasonality or False,
            seasonality_mode=prophet_config.seasonality_mode or "multiplicative",
            changepoint_prior_scale=prophet_config.changepoint_prior_scale or 0.05,
            holidays_prior_scale=prophet_config.holidays_prior_scale or 10.0,
            uncertainty_samples=prophet_config.uncertainty_samples or 1000
        )
        
        logger.info(f"Created Prophet model for {self.category}")
        return model
    
    def train(self, df: pd.DataFrame) -> Prophet:
        """
        Train Prophet model (Section 5.2.1)
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Trained Prophet model
        """
        if self.model is None:
            self.model = self.create_model()
        
        logger.info(f"Training Prophet model for {self.category} with {len(df)} records")
        self.model.fit(df)
        self.is_trained = True
        
        logger.info(f"Prophet model trained successfully for {self.category}")
        return self.model
    
    def predict(self, 
                periods: int = 30,
                freq: str = 'D') -> pd.DataFrame:
        """
        Generate forecasts (Section 5.2.1)
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with forecasts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        logger.info(f"Generated {periods} period forecast for {self.category}")
        return forecast
    
    def cross_validate(self, 
                      df: pd.DataFrame) -> Dict[str, float]:
        """
        Cross-validate model (Section 5.2.4)
        
        Args:
            df: Training data
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_trained:
            self.train(df)
        
        try:
            # Perform cross-validation
            df_cv = cross_validation(
                self.model,
                initial=self.config.training.cv_initial or "180 days",
                period=self.config.training.cv_period or "30 days",
                horizon=self.config.training.cv_horizon or "30 days",
                parallel="threads"
            )
            
            # Calculate performance metrics
            df_performance = performance_metrics(df_cv)
            
            # Extract key metrics
            metrics = {
                'rmse': df_performance['rmse'].mean(),
                'mae': df_performance['mae'].mean(),
                'mape': df_performance['mape'].mean(),
                'coverage': df_performance['coverage'].mean()
            }
            
            logger.info(f"Cross-validation metrics for {self.category}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Cross-validation failed for {self.category}: {e}", exc_info=True)
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'coverage': np.nan
            }
    
    def get_model_components(self, forecast: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract model components (Section 5.2.4)
        
        Args:
            forecast: Forecast DataFrame from predict()
            
        Returns:
            Dictionary with trend, seasonality components
        """
        components = {
            'trend': forecast[['ds', 'trend']],
            'yearly': forecast[['ds', 'yearly']] if 'yearly' in forecast.columns else None,
            'weekly': forecast[['ds', 'weekly']] if 'weekly' in forecast.columns else None,
            'holidays': forecast[['ds', 'holidays']] if 'holidays' in forecast.columns else None
        }
        
        return components
    
    def evaluate(self, 
                 forecast: pd.DataFrame,
                 actual: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            forecast: Forecast DataFrame
            actual: Actual values DataFrame with 'ds' and 'y' columns
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Merge forecast with actual
        merged = pd.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            actual[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        # Calculate metrics
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        
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


