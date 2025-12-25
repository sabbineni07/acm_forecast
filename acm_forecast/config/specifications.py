"""
Configuration Specifications for Azure Cost Management Forecasting Model
Section 7: Model Implementation - Configuration

All configuration dataclasses using Pydantic for validation and nested structure parsing
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from pydantic.dataclasses import dataclass
from pydantic import Field, ValidationError, TypeAdapter
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data source and processing configuration"""
    # Core required fields - must be provided
    delta_table_path: str = Field(..., description="Delta table path")
    database_name: str = Field(..., description="Database name")
    table_name: str = Field(..., description="Table name")
    
    # Optional fields with defaults in YAML
    min_historical_months: Optional[int] = Field(None, ge=1, description="Minimum historical months required")
    max_data_delay_hours: Optional[int] = Field(None, ge=0, description="Maximum data delay in hours")
    primary_region: Optional[str] = None
    primary_region_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Primary region weight (0-1)")
    secondary_region: Optional[str] = None
    secondary_region_weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Secondary region weight (0-1)")
    cost_categories: Optional[List[str]] = None
    
    # Sample data generation (for development/testing)
    generate_sample_data: Optional[bool] = Field(False, description="If true, generate sample data instead of loading from Delta table")
    sample_data_days: Optional[int] = Field(365, ge=1, description="Number of days of sample data to generate")
    sample_data_records_per_day: Optional[int] = Field(100, ge=1, description="Average number of records per day in sample data")
    sample_data_subscriptions: Optional[int] = Field(3, ge=1, description="Number of different subscriptions in sample data")
    sample_data_start_date: Optional[str] = Field(None, description="Start date for sample data generation (YYYY-MM-DD, defaults to days ago from today)")


@dataclass
class ProphetConfig:
    """Prophet model configuration"""
    yearly_seasonality: Optional[bool] = None
    weekly_seasonality: Optional[bool] = None
    daily_seasonality: Optional[bool] = None
    seasonality_mode: Optional[str] = Field(None, pattern="^(multiplicative|additive)$")
    changepoint_prior_scale: Optional[float] = Field(None, gt=0.0, description="Prophet changepoint prior scale")
    holidays_prior_scale: Optional[float] = Field(None, gt=0.0, description="Prophet holidays prior scale")
    uncertainty_samples: Optional[int] = Field(None, ge=100, description="Prophet uncertainty samples")


@dataclass
class ArimaConfig:
    """ARIMA model configuration"""
    seasonal: Optional[bool] = None
    seasonal_period: Optional[int] = Field(None, ge=1, description="ARIMA seasonal period")
    max_p: Optional[int] = Field(None, ge=1, le=10, description="ARIMA max p parameter")
    max_d: Optional[int] = Field(None, ge=1, le=5, description="ARIMA max d parameter")
    max_q: Optional[int] = Field(None, ge=1, le=10, description="ARIMA max q parameter")
    information_criterion: Optional[str] = Field(None, pattern="^(aic|bic)$", description="ARIMA information criterion")


@dataclass
class XGBoostConfig:
    """XGBoost model configuration"""
    n_estimators: Optional[int] = Field(None, ge=1, description="XGBoost number of estimators")
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="XGBoost max depth")
    learning_rate: Optional[float] = Field(None, gt=0.0, le=1.0, description="XGBoost learning rate")
    subsample: Optional[float] = Field(None, gt=0.0, le=1.0, description="XGBoost subsample ratio")
    colsample_bytree: Optional[float] = Field(None, gt=0.0, le=1.0, description="XGBoost column sample by tree")
    objective: Optional[str] = None
    early_stopping_rounds: Optional[int] = Field(None, ge=1, description="XGBoost early stopping rounds")


@dataclass
class ModelConfig:
    """Model configuration with nested sub-configs"""
    prophet: ProphetConfig
    arima: ArimaConfig
    xgboost: XGBoostConfig


@dataclass
class TrainingConfig:
    """Training and validation configuration"""
    # All optional with defaults in YAML
    train_split: Optional[float] = Field(None, gt=0.0, lt=1.0, description="Training split ratio")
    validation_split: Optional[float] = Field(None, gt=0.0, lt=1.0, description="Validation split ratio")
    test_split: Optional[float] = Field(None, gt=0.0, lt=1.0, description="Test split ratio")
    min_training_months: Optional[int] = Field(None, ge=1, description="Minimum training months")
    recommended_training_months: Optional[int] = Field(None, ge=1, description="Recommended training months")
    validation_months: Optional[int] = Field(None, ge=1, description="Validation months")
    test_months: Optional[int] = Field(None, ge=1, description="Test months")
    cv_initial: Optional[str] = None
    cv_period: Optional[str] = None
    cv_horizon: Optional[str] = None


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Core required fields
    target_column: str = Field(..., description="Target column name")
    date_column: str = Field(..., description="Date column name")
    
    # Optional fields with defaults in YAML
    lag_periods: Optional[List[int]] = Field(None, description="Lag periods for feature engineering")
    rolling_windows: Optional[List[int]] = Field(None, description="Rolling window sizes")


@dataclass
class PerformanceConfig:
    """Performance targets and thresholds"""
    # All optional with defaults in YAML
    target_mape: Optional[float] = Field(None, ge=0.0, description="Target MAPE percentage")
    target_r2: Optional[float] = Field(None, ge=0.0, le=1.0, description="Target R² score")
    warning_mape: Optional[float] = Field(None, ge=0.0, description="Warning MAPE threshold")
    critical_mape: Optional[float] = Field(None, ge=0.0, description="Critical MAPE threshold")
    warning_missing_data: Optional[float] = Field(None, ge=0.0, le=100.0, description="Warning missing data percentage")
    critical_missing_data: Optional[float] = Field(None, ge=0.0, le=100.0, description="Critical missing data percentage")


@dataclass
class RegistryConfig:
    """MLflow Model Registry configuration"""
    # Core required field
    mlflow_experiment_name: str = Field(..., description="MLflow experiment name")
    
    # Optional fields
    mlflow_tracking_uri: Optional[str] = None
    prophet_model_name: Optional[str] = None
    arima_model_name: Optional[str] = None
    xgboost_model_name: Optional[str] = None
    model_stages: Optional[List[str]] = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    # All optional with defaults in YAML
    monthly_retraining: Optional[bool] = None
    quarterly_retraining: Optional[bool] = None
    retraining_trigger_mape: Optional[float] = Field(None, ge=0.0, description="MAPE threshold to trigger retraining")
    max_months_without_retraining: Optional[int] = Field(None, ge=1, description="Max months without retraining")
    realtime_monitoring: Optional[bool] = None
    daily_reports: Optional[bool] = None
    weekly_reports: Optional[bool] = None
    monthly_reports: Optional[bool] = None


@dataclass
class ForecastConfig:
    """Forecast generation configuration"""
    # All optional with defaults in YAML
    forecast_horizons_days: Optional[List[int]] = Field(None, description="Forecast horizons in days")
    daily_forecasts: Optional[bool] = None
    weekly_forecasts: Optional[bool] = None
    monthly_forecasts: Optional[bool] = None


@dataclass
class AppConfig:
    """
    Main configuration class that aggregates all configs as a Pydantic dataclass
    Pydantic automatically handles nested structure parsing from YAML
    """
    name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    feature: FeatureConfig
    performance: PerformanceConfig
    registry: RegistryConfig
    monitoring: MonitoringConfig
    forecast: ForecastConfig
    
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> 'AppConfig':
        """
        Load configuration from YAML file and create AppConfig instance with validation
        Pydantic automatically parses the nested YAML structure
        
        Args:
            config_path: Path to YAML config file (default: acm_forecast/config/config.yaml)
            
        Returns:
            AppConfig instance with loaded and validated configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If validation fails (from Pydantic)
        """
        # Determine config path
        if config_path is None:
            config_dir = Path(__file__).parent
            config_path = config_dir / "config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            # Load YAML
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data is None:
                raise ValueError("YAML file is empty or invalid")
            
            logger.info(f"Loaded configuration from: {config_path}")
            
            # Handle mlflow_tracking_uri from env var if not in YAML
            registry_section = yaml_data.get('registry', {})
            if 'mlflow_tracking_uri' not in registry_section or registry_section.get('mlflow_tracking_uri') is None:
                env_uri = os.getenv("MLFLOW_TRACKING_URI")
                if env_uri:
                    yaml_data['registry'] = yaml_data.get('registry', {}).copy()
                    yaml_data['registry']['mlflow_tracking_uri'] = env_uri
            
            # Pydantic automatically handles nested structure parsing
            config = cls(**yaml_data)
            
            logger.info(f"Successfully loaded and validated configuration from: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert AppConfig to dictionary using Pydantic's TypeAdapter
        This leverages Pydantic's serialization which properly handles nested structures
        and respects validation rules
        """
        adapter = TypeAdapter(AppConfig)
        return adapter.dump_python(self)
    
    def to_yaml(self, output_path: Optional[str] = None, default_flow_style: bool = False) -> str:
        """
        Convert AppConfig to YAML string and optionally save to file
        
        Args:
            output_path: Optional path to save YAML file. If None, returns YAML string only.
            default_flow_style: Whether to use flow style for lists/dicts in YAML
            
        Returns:
            YAML string representation of the configuration
        """
        # Convert to dict using Pydantic's TypeAdapter (preserves nested structure)
        yaml_dict = self.to_dict()
        
        # Generate YAML string
        yaml_str = yaml.dump(
            yaml_dict,
            default_flow_style=default_flow_style,
            sort_keys=False,
            allow_unicode=True
        )
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Configuration saved to: {output_path}")
        
        return yaml_str
