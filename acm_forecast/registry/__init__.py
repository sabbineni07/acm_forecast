"""
Model Registry Module
Section 7.2: Model Registry Configuration in Databricks
Section 7.3: Access, Versioning and Controls Description

This module exports plugins directly. Use PluginFactory to create instances:
    from acm_forecast.core import PluginFactory
    factory = PluginFactory()
    registry = factory.create_model_registry(config, plugin_name="mlflow")
"""

from ..core import PluginFactory
from ..config import AppConfig
from typing import Optional

# Re-export plugin classes for convenience (users should use PluginFactory)
from ..plugins.model_registry.mlflow_registry import MLflowModelRegistry

# Export ModelVersioning (not a plugin, standalone class)
from .model_versioning import ModelVersioning

__all__ = [
    "MLflowModelRegistry",
    "ModelVersioning",
    "PluginFactory",  # Re-export for convenience
]

# Convenience function that uses PluginFactory
def create_model_registry(config: AppConfig, plugin_name: str = "mlflow", **kwargs):
    """Create a model registry plugin instance"""
    factory = PluginFactory()
    return factory.create_model_registry(config, plugin_name=plugin_name, **kwargs)
