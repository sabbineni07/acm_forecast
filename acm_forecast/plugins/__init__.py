"""
Built-in Plugins

This module exports all built-in plugin implementations.
Plugins are organized by type and automatically registered with the PluginRegistry.
"""

from typing import Dict, Type

from ..core.interfaces import (
    IDataSource,
    IDataQuality,
    IDataPreparation,
    IFeatureEngineer,
    IModel,
    IForecaster,
    IModelRegistry,
)

# Import built-in plugin implementations
from .data_source.delta_data_source import DeltaDataSource
from .data_quality.default_data_quality import DefaultDataQuality
from .data_preparation.default_data_preparation import DefaultDataPreparation
from .feature_engineer.default_feature_engineer import DefaultFeatureEngineer
from .models.prophet_model_plugin import ProphetModelPlugin
from .models.arima_model_plugin import ARIMAModelPlugin

from .forecasters.default_forecaster import DefaultForecaster
from .model_registry.mlflow_registry import MLflowModelRegistry

# Export plugin dictionaries for registration
data_source_plugins: Dict[str, Type[IDataSource]] = {
    'delta': DeltaDataSource,
}

data_quality_plugins: Dict[str, Type[IDataQuality]] = {
    'default': DefaultDataQuality,
}

data_preparation_plugins: Dict[str, Type[IDataPreparation]] = {
    'default': DefaultDataPreparation,
}

feature_engineer_plugins: Dict[str, Type[IFeatureEngineer]] = {
    'default': DefaultFeatureEngineer,
}

# Build model_plugins dictionary, conditionally including XGBoost
model_plugins: Dict[str, Type[IModel]] = {
    'prophet': ProphetModelPlugin,
    'arima': ARIMAModelPlugin,
}

# Lazy import for XGBoost (may fail if OpenMP runtime is not available)
try:
    from .models.xgboost_model_plugin import XGBoostModelPlugin
    model_plugins['xgboost'] = XGBoostModelPlugin
except Exception as e:
    # XGBoost not available (likely missing OpenMP runtime)
    # Framework will work without it - users just can't use xgboost plugin
    # Catch all exceptions since XGBoostError may not inherit from ImportError/OSError
    pass

forecaster_plugins: Dict[str, Type[IForecaster]] = {
    'default': DefaultForecaster,
}

model_registry_plugins: Dict[str, Type[IModelRegistry]] = {
    'mlflow': MLflowModelRegistry,
}

__all__ = [
    'data_source_plugins',
    'data_quality_plugins',
    'data_preparation_plugins',
    'feature_engineer_plugins',
    'model_plugins',
    'forecaster_plugins',
    'model_registry_plugins',
]

