"""
Plugin Registry System

Enables runtime plugin discovery, loading, and instantiation based on YAML configuration.
Supports both built-in plugins and custom user plugins.
"""

from typing import Dict, Any, Optional, Type, Callable
import importlib
import logging
from pathlib import Path

from .interfaces import (
    IDataSource,
    IDataQuality,
    IDataPreparation,
    IFeatureEngineer,
    IModel,
    IForecaster,
    IModelRegistry,
)

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        """Initialize plugin registry"""
        self._plugins: Dict[str, Dict[str, Type]] = {
            'data_source': {},
            'data_quality': {},
            'data_preparation': {},
            'feature_engineer': {},
            'model': {},
            'forecaster': {},
            'model_registry': {},
        }
        self._factories: Dict[str, Dict[str, Callable]] = {
            'data_source': {},
            'data_quality': {},
            'data_preparation': {},
            'feature_engineer': {},
            'model': {},
            'forecaster': {},
            'model_registry': {},
        }
        self._built_in_plugins_loaded = False
    
    def register(self, 
                 plugin_type: str,
                 name: str,
                 plugin_class: Type,
                 factory: Optional[Callable] = None):
        """Register a plugin class
        
        Args:
            plugin_type: Type of plugin (data_source, model, etc.)
            name: Plugin name (used in YAML config)
            plugin_class: Plugin class that implements the interface
            factory: Optional factory function for custom instantiation
        """
        if plugin_type not in self._plugins:
            raise ValueError(f"Unknown plugin type: {plugin_type}")
        
        self._plugins[plugin_type][name] = plugin_class
        if factory:
            self._factories[plugin_type][name] = factory
        else:
            # Default factory: instantiate with config and spark
            self._factories[plugin_type][name] = lambda cls, config, spark, **kwargs: cls(config, spark, **kwargs)
        
        logger.debug(f"Registered {plugin_type} plugin: {name}")
    
    def register_from_module(self, 
                            plugin_type: str,
                            name: str,
                            module_path: str,
                            class_name: str):
        """Register plugin from external module
        
        Args:
            plugin_type: Type of plugin
            name: Plugin name
            module_path: Full module path (e.g., 'my_plugins.custom_data_source')
            class_name: Class name within module
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            self.register(plugin_type, name, plugin_class)
            logger.info(f"Registered external {plugin_type} plugin: {name} from {module_path}")
        except Exception as e:
            logger.error(f"Failed to register plugin {name} from {module_path}: {e}")
            raise
    
    def load_built_in_plugins(self):
        """Load built-in plugins from acm_forecast.plugins"""
        if self._built_in_plugins_loaded:
            return
        
        try:
            from ..plugins import (
                data_source_plugins,
                data_quality_plugins,
                data_preparation_plugins,
                feature_engineer_plugins,
                model_plugins,
                forecaster_plugins,
                model_registry_plugins,
            )
            
            # Register all built-in plugins
            for name, cls in data_source_plugins.items():
                if cls is not None:  # Skip if plugin is None (e.g., XGBoost not available)
                    self.register('data_source', name, cls)
            
            for name, cls in data_quality_plugins.items():
                if cls is not None:
                    self.register('data_quality', name, cls)
            
            for name, cls in data_preparation_plugins.items():
                if cls is not None:
                    self.register('data_preparation', name, cls)
            
            for name, cls in feature_engineer_plugins.items():
                if cls is not None:
                    self.register('feature_engineer', name, cls)
            
            for name, cls in model_plugins.items():
                if cls is not None:  # Skip None values (e.g., XGBoost not available)
                    self.register('model', name, cls)
            
            for name, cls in forecaster_plugins.items():
                self.register('forecaster', name, cls)
            
            for name, cls in model_registry_plugins.items():
                self.register('model_registry', name, cls)
            
            self._built_in_plugins_loaded = True
            logger.info("Loaded all built-in plugins")
        except ImportError as e:
            logger.warning(f"Could not load built-in plugins: {e}")
    
    def get_plugin_class(self, plugin_type: str, name: str) -> Optional[Type]:
        """Get plugin class by type and name
        
        Args:
            plugin_type: Type of plugin
            name: Plugin name
            
        Returns:
            Plugin class or None if not found
        """
        if not self._built_in_plugins_loaded:
            self.load_built_in_plugins()
        
        return self._plugins.get(plugin_type, {}).get(name)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, list]:
        """List all registered plugins
        
        Args:
            plugin_type: Optional filter by plugin type
            
        Returns:
            Dictionary mapping plugin types to lists of plugin names
        """
        if not self._built_in_plugins_loaded:
            self.load_built_in_plugins()
        
        if plugin_type:
            return {plugin_type: list(self._plugins.get(plugin_type, {}).keys())}
        
        return {pt: list(plugins.keys()) for pt, plugins in self._plugins.items()}


class PluginFactory:
    """Factory for creating plugin instances from configuration"""
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize plugin factory
        
        Args:
            registry: Optional PluginRegistry instance (creates new if not provided)
        """
        self.registry = registry or PluginRegistry()
        self.registry.load_built_in_plugins()
    
    def create_data_source(self, 
                          config,
                          spark=None,
                          plugin_name: Optional[str] = None,
                          **kwargs) -> IDataSource:
        """Create data source plugin instance
        
        Args:
            config: AppConfig instance
            spark: Optional SparkSession
            plugin_name: Plugin name from config (defaults to config.plugins.data_source.name)
            **kwargs: Additional plugin configuration
            
        Returns:
            IDataSource instance
        """
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and config.plugins.data_source:
                plugin_name = config.plugins.data_source.name
            else:
                plugin_name = 'acm'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('data_source', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Data source plugin '{plugin_name}' not found. Available: {list(self.registry._plugins['data_source'].keys())}")
        
        # Get plugin-specific config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and config.plugins.data_source:
            plugin_config = config.plugins.data_source.config or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['data_source'].get(plugin_name)
        return factory(plugin_class, config, spark, **kwargs)
    
    def create_data_quality(self,
                           config,
                           spark=None,
                           plugin_name: Optional[str] = None,
                           **kwargs) -> IDataQuality:
        """Create data quality plugin instance"""
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'data_quality') and config.plugins.data_quality:
                plugin_name = config.plugins.data_quality.name
            else:
                plugin_name = 'default'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('data_quality', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Data quality plugin '{plugin_name}' not found")
        
        # Get plugin-specific config from config.plugins.data_quality.config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'data_quality') and config.plugins.data_quality:
            plugin_config = getattr(config.plugins.data_quality, 'config', {}) or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['data_quality'].get(plugin_name)
        return factory(plugin_class, config, spark, **kwargs)
    
    def create_data_preparation(self,
                               config,
                               spark=None,
                               plugin_name: Optional[str] = None,
                               **kwargs) -> IDataPreparation:
        """Create data preparation plugin instance"""
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'data_preparation') and config.plugins.data_preparation:
                plugin_name = config.plugins.data_preparation.name
            else:
                plugin_name = 'acm'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('data_preparation', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Data preparation plugin '{plugin_name}' not found")
        
        # Get plugin-specific config from config.plugins.data_preparation.config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'data_preparation') and config.plugins.data_preparation:
            plugin_config = getattr(config.plugins.data_preparation, 'config', {}) or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['data_preparation'].get(plugin_name)
        return factory(plugin_class, config, spark, **kwargs)
    
    def create_feature_engineer(self,
                               config,
                               spark=None,
                               plugin_name: Optional[str] = None,
                               **kwargs) -> IFeatureEngineer:
        """Create feature engineer plugin instance"""
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'feature_engineer') and config.plugins.feature_engineer:
                plugin_name = config.plugins.feature_engineer.name
            else:
                plugin_name = 'default'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('feature_engineer', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Feature engineer plugin '{plugin_name}' not found")
        
        # Get plugin-specific config from config.plugins.feature_engineer.config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'feature_engineer') and config.plugins.feature_engineer:
            plugin_config = getattr(config.plugins.feature_engineer, 'config', {}) or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['feature_engineer'].get(plugin_name)
        return factory(plugin_class, config, spark, **kwargs)
    
    def create_model(self,
                    config,
                    category: str = "Total",
                    plugin_name: Optional[str] = None,
                    **kwargs) -> IModel:
        """Create model plugin instance"""
        # Get model name from config if not provided
        if plugin_name is None:
            model_type = config.model.selected_model
            plugin_name = model_type
        
        plugin_class = self.registry.get_plugin_class('model', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Model plugin '{plugin_name}' not found")
        
        # Get model-specific config from config.model
        model_config = getattr(config.model, plugin_name, {})
        plugin_config = kwargs.get('plugin_config', {})
        if hasattr(model_config, '__dict__'):
            plugin_config.update(model_config.__dict__)
        
        factory = self.registry._factories['model'].get(plugin_name)
        return factory(plugin_class, config, category, plugin_config=plugin_config)
    
    def create_forecaster(self,
                         config,
                         plugin_name: Optional[str] = None,
                         **kwargs) -> IForecaster:
        """Create forecaster plugin instance"""
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'forecaster') and config.plugins.forecaster:
                plugin_name = config.plugins.forecaster.name
            else:
                plugin_name = 'default'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('forecaster', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Forecaster plugin '{plugin_name}' not found")
        
        # Get plugin-specific config from config.plugins.forecaster.config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'forecaster') and config.plugins.forecaster:
            plugin_config = getattr(config.plugins.forecaster, 'config', {}) or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['forecaster'].get(plugin_name)
        return factory(plugin_class, config, **kwargs)
    
    def create_model_registry(self,
                             config,
                             plugin_name: Optional[str] = None,
                             **kwargs) -> IModelRegistry:
        """Create model registry plugin instance"""
        # Get plugin name from config or use default
        if plugin_name is None:
            if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'model_registry') and config.plugins.model_registry:
                plugin_name = config.plugins.model_registry.name
            else:
                plugin_name = 'mlflow'  # Default built-in plugin
        
        plugin_class = self.registry.get_plugin_class('model_registry', plugin_name)
        
        if plugin_class is None:
            raise ValueError(f"Model registry plugin '{plugin_name}' not found")
        
        # Get plugin-specific config from config.plugins.model_registry.config
        plugin_config = {}
        if hasattr(config, 'plugins') and config.plugins and hasattr(config.plugins, 'model_registry') and config.plugins.model_registry:
            plugin_config = getattr(config.plugins.model_registry, 'config', {}) or {}
        plugin_config.update(kwargs.pop('plugin_config', {}))
        kwargs['plugin_config'] = plugin_config
        
        factory = self.registry._factories['model_registry'].get(plugin_name)
        return factory(plugin_class, config, **kwargs)

