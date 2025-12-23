"""
Integration tests for configuration and pipeline integration
"""
import pytest

from acm_forecast.config import AppConfig


class TestConfigPipelineIntegration:
    """Integration tests for config loading and pipeline usage"""
    
    @pytest.mark.integration
    def test_config_loads_and_validates(self, temp_config_file):
        """Test that config loads and validates correctly"""
        config = AppConfig.from_yaml(temp_config_file)
        
        # Verify all required sections exist
        assert config.name is not None
        assert config.data is not None
        assert config.model is not None
        assert config.training is not None
        assert config.feature is not None
        assert config.performance is not None
        assert config.registry is not None
        assert config.monitoring is not None
        assert config.forecast is not None
    
    @pytest.mark.integration
    def test_config_serialization_roundtrip(self, temp_config_file):
        """Test that config can be serialized and deserialized"""
        original_config = AppConfig.from_yaml(temp_config_file)
        
        # Convert to dict and back (simulating serialization)
        config_dict = original_config.to_dict()
        
        # Create new config from dict
        reconstructed_config = AppConfig(**config_dict)
        
        # Verify key fields match
        assert reconstructed_config.name == original_config.name
        assert reconstructed_config.data.delta_table_path == original_config.data.delta_table_path
        assert reconstructed_config.model.prophet.yearly_seasonality == original_config.model.prophet.yearly_seasonality
    
    @pytest.mark.integration
    def test_config_yaml_roundtrip(self, temp_config_file):
        """Test that config can be saved to YAML and loaded back"""
        import tempfile
        import os
        
        original_config = AppConfig.from_yaml(temp_config_file)
        
        # Save to new YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_output = f.name
        
        try:
            original_config.to_yaml(output_path=temp_output)
            
            # Load back
            loaded_config = AppConfig.from_yaml(temp_output)
            
            # Verify key fields match
            assert loaded_config.name == original_config.name
            assert loaded_config.data.database_name == original_config.data.database_name
            assert loaded_config.feature.target_column == original_config.feature.target_column
        finally:
            if os.path.exists(temp_output):
                os.unlink(temp_output)
    
    @pytest.mark.integration
    def test_config_nested_structure_access(self, sample_app_config):
        """Test accessing nested configuration structure"""
        # Test deeply nested access
        assert sample_app_config.model.prophet.yearly_seasonality is not None
        assert sample_app_config.model.arima.seasonal_period is not None
        assert sample_app_config.model.xgboost.n_estimators is not None
        
        # Test all config sections are accessible
        assert hasattr(sample_app_config, 'data')
        assert hasattr(sample_app_config, 'model')
        assert hasattr(sample_app_config, 'training')
        assert hasattr(sample_app_config, 'feature')
        assert hasattr(sample_app_config, 'performance')
        assert hasattr(sample_app_config, 'registry')
        assert hasattr(sample_app_config, 'monitoring')
        assert hasattr(sample_app_config, 'forecast')

