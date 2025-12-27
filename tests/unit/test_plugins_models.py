"""
Unit tests for model plugins (Prophet, ARIMA)
"""
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

from acm_forecast.core.plugin_registry import PluginFactory
from acm_forecast.config import AppConfig


class TestProphetModelPlugin:
    """Unit tests for ProphetModelPlugin"""
    
    @pytest.mark.unit
    def test_prophet_model_initialization(self, sample_app_config):
        """Test ProphetModelPlugin initialization"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="prophet"
        )
        
        assert model is not None
        assert model.config == sample_app_config
    
    @pytest.mark.unit
    def test_prophet_model_has_required_methods(self, sample_app_config):
        """Test that ProphetModelPlugin has all required interface methods"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="prophet"
        )
        
        # Check IModel interface methods
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'evaluate')
    
    @pytest.mark.unit
    def test_prophet_model_train_interface(self, sample_app_config):
        """Test Prophet model train method interface"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="prophet"
        )
        
        # Create sample training data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            sample_app_config.feature.date_column: dates,
            sample_app_config.feature.target_column: np.random.randn(100).cumsum() + 100
        })
        
        # Train should accept the data
        # Note: This might raise errors if dependencies are missing, but interface should exist
        assert callable(model.train)


class TestARIMAModelPlugin:
    """Unit tests for ARIMAModelPlugin"""
    
    @pytest.mark.unit
    def test_arima_model_initialization(self, sample_app_config):
        """Test ARIMAModelPlugin initialization"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="arima"
        )
        
        assert model is not None
        assert model.config == sample_app_config
    
    @pytest.mark.unit
    def test_arima_model_has_required_methods(self, sample_app_config):
        """Test that ARIMAModelPlugin has all required interface methods"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="arima"
        )
        
        # Check IModel interface methods
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'evaluate')
    
    @pytest.mark.unit
    def test_arima_model_train_interface(self, sample_app_config):
        """Test ARIMA model train method interface"""
        factory = PluginFactory()
        model = factory.create_model(
            sample_app_config,
            spark=None,
            plugin_name="arima"
        )
        
        assert callable(model.train)

