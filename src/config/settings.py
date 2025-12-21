"""
Configuration settings for Azure Cost Management Forecasting Model
Section 7: Model Implementation - Configuration
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Data source and processing configuration (Section 3.1)"""
    # Data source location
    delta_table_path: str = os.getenv(
        "DELTA_TABLE_PATH",
        "azure_cost_management.amortized_costs"
    )
    database_name: str = os.getenv("DATABASE_NAME", "cost_management")
    table_name: str = os.getenv("TABLE_NAME", "amortized_costs")
    
    # Data constraints
    min_historical_months: int = 12
    max_data_delay_hours: int = 168 # 48
    
    # Regional distribution (Section 3.3.1)
    primary_region: str = "East US"
    primary_region_weight: float = 0.9
    secondary_region: str = "South Central US"
    secondary_region_weight: float = 0.1
    
    # Cost categories (Section 3.3.5)
    cost_categories: List[str] = None
    
    def __post_init__(self):
        if self.cost_categories is None:
            self.cost_categories = [
                "Compute", "Storage", "Network", "Database",
                "Analytics", "AI/ML", "Security", "Management"
            ]


@dataclass
class ModelConfig:
    """Model configuration (Section 5.2.3)"""
    
    # Prophet configuration
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    prophet_seasonality_mode: str = "multiplicative"
    prophet_changepoint_prior_scale: float = 0.05
    prophet_holidays_prior_scale: float = 10.0
    prophet_uncertainty_samples: int = 1000
    
    # ARIMA configuration
    arima_seasonal: bool = True
    arima_seasonal_period: int = 12  # Monthly
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    arima_information_criterion: str = "aic"
    
    # XGBoost configuration
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.8
    xgboost_objective: str = "reg:squarederror"
    xgboost_early_stopping_rounds: int = 10


@dataclass
class TrainingConfig:
    """Training and validation configuration (Section 3.3.2)"""
    # Data sampling
    train_split: float = 0.70  # 70% for training
    validation_split: float = 0.15  # 15% for validation
    test_split: float = 0.15  # 15% for testing
    
    # Minimum data requirements
    min_training_months: int = 12
    recommended_training_months: int = 18
    validation_months: int = 3
    test_months: int = 3
    
    # Cross-validation
    cv_initial: str = "180 days"
    cv_period: str = "30 days"
    cv_horizon: str = "30 days"


@dataclass
class FeatureConfig:
    """Feature engineering configuration (Section 3.3.4, 5.1)"""
    # Lag features
    lag_periods: List[int] = None
    
    # Rolling window features
    rolling_windows: List[int] = None
    
    # Target variable
    target_column: str = "PreTaxCost"
    date_column: str = "UsageDateTime"
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 7, 14, 30]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 30]


@dataclass
class PerformanceConfig:
    """Performance targets and thresholds (Section 6.1, 8.1)"""
    # Target metrics
    target_mape: float = 10.0  # < 10% MAPE for monthly forecasts
    target_r2: float = 0.8  # > 0.8 R²
    
    # Alert thresholds
    warning_mape: float = 12.0  # Warning if MAPE > 12%
    critical_mape: float = 15.0  # Critical if MAPE > 15%
    
    # Data quality thresholds
    warning_missing_data: float = 5.0  # Warning if > 5% missing
    critical_missing_data: float = 10.0  # Critical if > 10% missing


@dataclass
class RegistryConfig:
    """MLflow Model Registry configuration (Section 7.2)"""
    # MLflow settings
    mlflow_tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = "/Users/team/azure_cost_forecasting"
    
    # Model names
    prophet_model_name: str = "azure_cost_forecast_prophet"
    arima_model_name: str = "azure_cost_forecast_arima"
    xgboost_model_name: str = "azure_cost_forecast_xgboost"
    
    # Model stages
    model_stages: List[str] = None
    
    def __post_init__(self):
        if self.model_stages is None:
            self.model_stages = ["None", "Staging", "Production"]


@dataclass
class MonitoringConfig:
    """Monitoring configuration (Section 8)"""
    # Retraining schedule
    monthly_retraining: bool = True
    quarterly_retraining: bool = True
    retraining_trigger_mape: float = 15.0  # Retrain if MAPE > 15%
    max_months_without_retraining: int = 6
    
    # Monitoring frequency
    realtime_monitoring: bool = True
    daily_reports: bool = True
    weekly_reports: bool = True
    monthly_reports: bool = True


@dataclass
class ForecastConfig:
    """Forecast generation configuration (Section 2.1)"""
    # Forecast horizons
    forecast_horizons_days: List[int] = None
    
    # Granularity
    daily_forecasts: bool = True
    weekly_forecasts: bool = True
    monthly_forecasts: bool = True
    
    def __post_init__(self):
        if self.forecast_horizons_days is None:
            self.forecast_horizons_days = [30, 90, 180, 365]  # 1, 3, 6, 12 months


# Global configuration instances
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
feature_config = FeatureConfig()
performance_config = PerformanceConfig()
registry_config = RegistryConfig()
monitoring_config = MonitoringConfig()
forecast_config = ForecastConfig()


