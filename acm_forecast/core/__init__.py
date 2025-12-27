"""
Core Framework Interfaces and Plugin System

This module provides the foundation for the pluggable framework architecture:
- Abstract interfaces for all pluggable components
- Plugin registry for runtime plugin discovery and loading
- Factory pattern for creating plugin instances
"""

from .interfaces import (
    IDataSource,
    IDataQuality,
    IDataPreparation,
    IFeatureEngineer,
    IModel,
    IForecaster,
    IModelRegistry,
)

from .plugin_registry import PluginRegistry, PluginFactory
from .base_plugin import BasePlugin

__all__ = [
    "IDataSource",
    "IDataQuality",
    "IDataPreparation",
    "IFeatureEngineer",
    "IModel",
    "IForecaster",
    "IModelRegistry",
    "PluginRegistry",
    "PluginFactory",
    "BasePlugin",
]

