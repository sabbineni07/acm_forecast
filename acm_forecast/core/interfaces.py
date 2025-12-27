"""
Abstract Interfaces for Pluggable Components

All components in the framework implement these interfaces, enabling:
- Dependency injection
- Plugin-based architecture
- Runtime component swapping
- Custom implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pyspark.sql import DataFrame, SparkSession
import pandas as pd

from ..config import AppConfig


class IDataSource(ABC):
    """Interface for data source implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize data source
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def load_data(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  **filters) -> DataFrame:
        """Load data from source
        
        Args:
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            **filters: Additional filters (category, region, etc.)
            
        Returns:
            Spark DataFrame with loaded data
        """
        pass
    
    @abstractmethod
    def get_data_profile(self, df: DataFrame) -> Dict[str, Any]:
        """Get data profile information
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with profile information
        """
        pass
    
    @abstractmethod
    def validate_data_availability(self, df: DataFrame) -> Dict[str, Any]:
        """Validate data availability and constraints
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation results dictionary
        """
        pass


class IDataQuality(ABC):
    """Interface for data quality validation implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize data quality validator
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def validate_completeness(self, df: DataFrame) -> Dict[str, Any]:
        """Validate data completeness
        
        Args:
            df: Input DataFrame
            
        Returns:
            Completeness validation results
        """
        pass
    
    @abstractmethod
    def validate_accuracy(self, df: DataFrame) -> Dict[str, Any]:
        """Validate data accuracy
        
        Args:
            df: Input DataFrame
            
        Returns:
            Accuracy validation results
        """
        pass
    
    @abstractmethod
    def validate_consistency(self, df: DataFrame) -> Dict[str, Any]:
        """Validate data consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            Consistency validation results
        """
        pass
    
    @abstractmethod
    def comprehensive_validation(self, df: DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality validation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Comprehensive validation results with quality score
        """
        pass


class IDataPreparation(ABC):
    """Interface for data preparation implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize data preparation
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def aggregate_data(self, df: DataFrame, group_by: Optional[List[str]] = None) -> DataFrame:
        """Aggregate data
        
        Args:
            df: Input DataFrame
            group_by: Optional list of columns to group by
            
        Returns:
            Aggregated DataFrame
        """
        pass
    
    @abstractmethod
    def prepare_for_training(self, df: DataFrame, model_type: str = "prophet") -> pd.DataFrame:
        """Prepare data for model training
        
        Args:
            df: Input Spark DataFrame
            model_type: Type of model (prophet, arima, xgboost)
            
        Returns:
            Prepared pandas DataFrame (converted from Spark DataFrame internally)
        """
        pass
    
    @abstractmethod
    def split(self, df: DataFrame, date_col: Optional[str] = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Split data into train, validation, and test sets
        
        Args:
            df: Input Spark DataFrame (must be sorted by date)
            date_col: Date column name (optional, uses config if not provided)
            
        Returns:
            Tuple of (train, validation, test) Spark DataFrames
        """
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize feature engineer
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        pass
    
    @abstractmethod
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        pass
    
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame, model_type: str = "xgboost") -> pd.DataFrame:
        """Prepare features for model training
        
        Args:
            df: Input DataFrame
            model_type: Type of model
            
        Returns:
            DataFrame with prepared features
        """
        pass


class IModel(ABC):
    """Interface for model implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, category: str = "Total", **kwargs):
        """Initialize model
        
        Args:
            config: AppConfig instance
            category: Category name
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the model
        
        Args:
            df: Training DataFrame
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, periods: int, **kwargs) -> pd.DataFrame:
        """Generate predictions
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional prediction parameters
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to path
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path
        
        Args:
            path: Path to load model from
        """
        pass


class IForecaster(ABC):
    """Interface for forecasting implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, **kwargs):
        """Initialize forecaster
        
        Args:
            config: AppConfig instance
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def generate_forecast(self, 
                         model_name: str,
                         horizon_days: int,
                         **kwargs) -> Dict[str, Any]:
        """Generate forecast
        
        Args:
            model_name: Name of model to use
            horizon_days: Forecast horizon in days
            **kwargs: Additional forecast parameters
            
        Returns:
            Forecast results dictionary
        """
        pass


class IModelRegistry(ABC):
    """Interface for model registry implementations"""
    
    @abstractmethod
    def __init__(self, config: AppConfig, **kwargs):
        """Initialize model registry
        
        Args:
            config: AppConfig instance
            **kwargs: Additional plugin-specific configuration
        """
        pass
    
    @abstractmethod
    def save_model(self, model: Any, name: str, version: Optional[str] = None) -> str:
        """Save model to registry
        
        Args:
            model: Model object to save
            name: Model name
            version: Optional version string
            
        Returns:
            Model version string
        """
        pass
    
    @abstractmethod
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """Load model from registry
        
        Args:
            name: Model name
            version: Optional version string (loads latest if not provided)
            
        Returns:
            Loaded model object
        """
        pass

