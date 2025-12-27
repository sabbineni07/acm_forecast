"""
Base Plugin Class

Provides common functionality for all plugins.
"""

from abc import ABC
from typing import Optional, Dict, Any
from pyspark.sql import SparkSession

from ..config import AppConfig


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize base plugin
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            **kwargs: Plugin-specific configuration
        """
        self.config = config
        self.spark = spark
        self.plugin_config = kwargs.get('plugin_config', {})
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get plugin-specific configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.plugin_config.get(key, default)
    
    def validate_spark(self) -> SparkSession:
        """Validate SparkSession is available
        
        Returns:
            SparkSession instance
            
        Raises:
            ValueError: If SparkSession is not available
        """
        if self.spark is None:
            raise ValueError(f"{self.__class__.__name__} requires SparkSession")
        return self.spark

