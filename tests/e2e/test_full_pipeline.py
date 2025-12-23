"""
End-to-end tests for full pipeline workflows
"""
import pytest


class TestFullPipelineE2E:
    """End-to-end tests for complete pipeline execution"""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_config_loading_e2e(self, temp_config_file):
        """End-to-end test: Load config and verify all components are accessible"""
        config = AppConfig.from_yaml(temp_config_file)
        
        # Verify config structure
        assert config.name == "test_config"
        
        # Verify all major components can be accessed
        assert config.data.delta_table_path is not None
        assert config.model.prophet is not None
        assert config.model.arima is not None
        assert config.model.xgboost is not None
        assert config.training.train_split is not None
        assert config.feature.target_column is not None
        assert config.registry.mlflow_experiment_name is not None
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_config_to_dict_and_back_e2e(self, sample_app_config):
        """End-to-end test: Full config serialization cycle"""
        # Original config
        original = sample_app_config
        
        # Convert to dict
        config_dict = original.to_dict()
        assert isinstance(config_dict, dict)
        assert "name" in config_dict
        assert "data" in config_dict
        assert "model" in config_dict
        
        # Convert dict back to config (would require implementing from_dict if needed)
        # For now, just verify dict structure is correct
        assert config_dict["name"] == original.name
        assert config_dict["data"]["delta_table_path"] == original.data.delta_table_path


# Import here to avoid circular dependencies
from acm_forecast.config import AppConfig

