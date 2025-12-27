"""
Unit tests for PluginRegistry
"""
import pytest
from typing import Type
from unittest.mock import Mock, MagicMock

from acm_forecast.core.plugin_registry import PluginRegistry
from acm_forecast.core.interfaces import IDataSource, IDataQuality, IModel


class TestPluginRegistry:
    """Unit tests for PluginRegistry"""
    
    @pytest.mark.unit
    def test_plugin_registry_initialization(self):
        """Test that PluginRegistry initializes correctly"""
        registry = PluginRegistry()
        assert registry._plugins is not None
        assert 'data_source' in registry._plugins
        assert 'model' in registry._plugins
        assert 'data_quality' in registry._plugins
        assert not registry._built_in_plugins_loaded
    
    @pytest.mark.unit
    def test_register_plugin(self):
        """Test registering a plugin"""
        registry = PluginRegistry()
        
        # Create a mock plugin class
        class MockDataSource(IDataSource):
            def load_data(self, **kwargs):
                pass
        
        registry.register('data_source', 'mock', MockDataSource)
        
        assert 'mock' in registry._plugins['data_source']
        assert registry._plugins['data_source']['mock'] == MockDataSource
    
    @pytest.mark.unit
    def test_register_invalid_plugin_type(self):
        """Test that registering with invalid plugin type raises error"""
        registry = PluginRegistry()
        
        class MockPlugin:
            pass
        
        with pytest.raises(ValueError, match="Unknown plugin type"):
            registry.register('invalid_type', 'test', MockPlugin)
    
    @pytest.mark.unit
    def test_get_plugin_class(self):
        """Test getting a plugin class"""
        registry = PluginRegistry()
        
        class MockDataSource(IDataSource):
            def load_data(self, **kwargs):
                pass
        
        registry.register('data_source', 'mock', MockDataSource)
        
        plugin_class = registry.get_plugin_class('data_source', 'mock')
        assert plugin_class == MockDataSource
    
    @pytest.mark.unit
    def test_get_plugin_class_not_found(self):
        """Test getting a plugin class that doesn't exist"""
        registry = PluginRegistry()
        
        plugin_class = registry.get_plugin_class('data_source', 'nonexistent')
        assert plugin_class is None
    
    @pytest.mark.unit
    def test_list_plugins(self):
        """Test listing plugins"""
        registry = PluginRegistry()
        
        class MockDataSource(IDataSource):
            def load_data(self, **kwargs):
                pass
        
        class MockDataSource2(IDataSource):
            def load_data(self, **kwargs):
                pass
        
        registry.register('data_source', 'mock1', MockDataSource)
        registry.register('data_source', 'mock2', MockDataSource2)
        
        # List all plugins (loads built-in plugins)
        all_plugins = registry.list_plugins()
        assert 'data_source' in all_plugins
        # Should include our registered plugins plus built-ins
        assert 'mock1' in all_plugins['data_source'] or len(all_plugins['data_source']) >= 2
        
        # List specific plugin type
        data_source_plugins = registry.list_plugins('data_source')
        assert 'data_source' in data_source_plugins
        assert 'mock1' in data_source_plugins['data_source']
        assert 'mock2' in data_source_plugins['data_source']
    
    @pytest.mark.unit
    def test_list_plugins_empty(self):
        """Test listing plugins when none are registered (except built-ins)"""
        registry = PluginRegistry()
        
        plugins = registry.list_plugins('data_source')
        assert 'data_source' in plugins
        # Even with no custom plugins, built-ins may be loaded
        assert isinstance(plugins['data_source'], list)
    
    @pytest.mark.unit
    def test_load_built_in_plugins(self):
        """Test loading built-in plugins"""
        registry = PluginRegistry()
        
        # This should load all built-in plugins
        registry.load_built_in_plugins()
        
        assert registry._built_in_plugins_loaded
        
        # Check that some plugins are registered
        data_source_plugins = registry.list_plugins('data_source')
        assert len(data_source_plugins['data_source']) > 0
    
    @pytest.mark.unit
    def test_load_built_in_plugins_idempotent(self):
        """Test that loading built-in plugins is idempotent"""
        registry = PluginRegistry()
        
        registry.load_built_in_plugins()
        first_count = len(registry.list_plugins()['data_source'])
        
        registry.load_built_in_plugins()
        second_count = len(registry.list_plugins()['data_source'])
        
        assert first_count == second_count
    
    @pytest.mark.unit
    def test_register_from_module(self):
        """Test registering a plugin from an external module"""
        registry = PluginRegistry()
        
        # This will fail if the module doesn't exist, which is expected
        with pytest.raises((ImportError, AttributeError)):
            registry.register_from_module(
                'data_source',
                'external',
                'nonexistent.module.path',
                'SomeClass'
            )

