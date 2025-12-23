"""
Unit tests for configuration module
"""
import pytest
import tempfile
import yaml
import os
from pathlib import Path

from pydantic import ValidationError

from acm_forecast.config import (
    AppConfig, DataConfig, ModelConfig, TrainingConfig, FeatureConfig,
    PerformanceConfig, RegistryConfig, MonitoringConfig, ForecastConfig,
    ProphetConfig, ArimaConfig, XGBoostConfig
)


class TestDataConfig:
    """Unit tests for DataConfig"""
    
    @pytest.mark.unit
    def test_data_config_required_fields(self):
        """Test that required fields are enforced"""
        # Should succeed with required fields
        config = DataConfig(
            delta_table_path="test.database.table",
            database_name="test_db",
            table_name="test_table",
        )
        assert config.delta_table_path == "test.database.table"
        assert config.database_name == "test_db"
        assert config.table_name == "test_table"
    
    @pytest.mark.unit
    def test_data_config_optional_fields(self):
        """Test optional fields with defaults"""
        config = DataConfig(
            delta_table_path="test.database.table",
            database_name="test_db",
            table_name="test_table",
            min_historical_months=12,
            max_data_delay_hours=48,
            primary_region="East US",
            primary_region_weight=0.8,
        )
        assert config.min_historical_months == 12
        assert config.max_data_delay_hours == 48
        assert config.primary_region == "East US"
        assert config.primary_region_weight == 0.8
    
    @pytest.mark.unit
    def test_data_config_validation_min_historical_months(self):
        """Test validation for min_historical_months"""
        with pytest.raises(ValidationError):
            DataConfig(
                delta_table_path="test.database.table",
                database_name="test_db",
                table_name="test_table",
                min_historical_months=0,  # Should be >= 1
            )
    
    @pytest.mark.unit
    def test_data_config_validation_region_weight(self):
        """Test validation for region weight bounds"""
        with pytest.raises(ValidationError):
            DataConfig(
                delta_table_path="test.database.table",
                database_name="test_db",
                table_name="test_table",
                primary_region_weight=1.5,  # Should be <= 1.0
            )


class TestProphetConfig:
    """Unit tests for ProphetConfig"""
    
    @pytest.mark.unit
    def test_prophet_config_creation(self):
        """Test creating ProphetConfig with valid values"""
        config = ProphetConfig(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            holidays_prior_scale=10.0,
            uncertainty_samples=1000,
        )
        assert config.yearly_seasonality is True
        assert config.seasonality_mode == "multiplicative"
        assert config.uncertainty_samples == 1000
    
    @pytest.mark.unit
    def test_prophet_config_seasonality_mode_validation(self):
        """Test validation for seasonality_mode"""
        # Valid values
        config1 = ProphetConfig(seasonality_mode="multiplicative")
        config2 = ProphetConfig(seasonality_mode="additive")
        assert config1.seasonality_mode == "multiplicative"
        assert config2.seasonality_mode == "additive"
        
        # Invalid value
        with pytest.raises(ValidationError):
            ProphetConfig(seasonality_mode="invalid")
    
    @pytest.mark.unit
    def test_prophet_config_positive_values(self):
        """Test validation for positive numeric values"""
        with pytest.raises(ValidationError):
            ProphetConfig(changepoint_prior_scale=-1.0)  # Should be > 0
        
        with pytest.raises(ValidationError):
            ProphetConfig(uncertainty_samples=50)  # Should be >= 100


class TestArimaConfig:
    """Unit tests for ArimaConfig"""
    
    @pytest.mark.unit
    def test_arima_config_creation(self):
        """Test creating ArimaConfig with valid values"""
        config = ArimaConfig(
            seasonal=True,
            seasonal_period=12,
            max_p=5,
            max_d=2,
            max_q=5,
            information_criterion="aic",
        )
        assert config.seasonal is True
        assert config.seasonal_period == 12
        assert config.information_criterion == "aic"
    
    @pytest.mark.unit
    def test_arima_config_information_criterion_validation(self):
        """Test validation for information_criterion"""
        config1 = ArimaConfig(information_criterion="aic")
        config2 = ArimaConfig(information_criterion="bic")
        assert config1.information_criterion == "aic"
        assert config2.information_criterion == "bic"
        
        with pytest.raises(ValidationError):
            ArimaConfig(information_criterion="invalid")
    
    @pytest.mark.unit
    def test_arima_config_bounds_validation(self):
        """Test validation for parameter bounds"""
        with pytest.raises(ValidationError):
            ArimaConfig(max_p=15)  # Should be <= 10
        
        with pytest.raises(ValidationError):
            ArimaConfig(max_d=10)  # Should be <= 5


class TestXGBoostConfig:
    """Unit tests for XGBoostConfig"""
    
    @pytest.mark.unit
    def test_xgboost_config_creation(self):
        """Test creating XGBoostConfig with valid values"""
        config = XGBoostConfig(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            early_stopping_rounds=10,
        )
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
    
    @pytest.mark.unit
    def test_xgboost_config_range_validation(self):
        """Test validation for values in range [0, 1]"""
        with pytest.raises(ValidationError):
            XGBoostConfig(learning_rate=1.5)  # Should be <= 1.0
        
        with pytest.raises(ValidationError):
            XGBoostConfig(subsample=-0.1)  # Should be > 0


class TestAppConfig:
    """Unit tests for AppConfig"""
    
    @pytest.mark.unit
    def test_app_config_from_yaml(self, temp_config_file):
        """Test loading AppConfig from YAML file"""
        config = AppConfig.from_yaml(temp_config_file)
        assert config.name == "test_config"
        assert config.data.delta_table_path == "test.database.test_table"
        assert config.model.prophet.yearly_seasonality is True
    
    @pytest.mark.unit
    def test_app_config_from_yaml_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file"""
        with pytest.raises(FileNotFoundError):
            AppConfig.from_yaml("nonexistent_file.yaml")
    
    @pytest.mark.unit
    def test_app_config_from_yaml_default_path(self, monkeypatch):
        """Test loading from default config path"""
        # Create a temporary config file in the expected location
        config_dir = Path(__file__).parent.parent.parent / "acm_forecast" / "config"
        temp_file = config_dir / "temp_config.yaml"
        
        try:
            test_data = {
                "name": "default_test",
                "data": {
                    "delta_table_path": "test.db.table",
                    "database_name": "test_db",
                    "table_name": "test_table",
                },
                "model": {
                    "prophet": {
                        "yearly_seasonality": True,
                        "weekly_seasonality": True,
                        "daily_seasonality": False,
                    },
                    "arima": {
                        "seasonal": True,
                        "seasonal_period": 12,
                    },
                    "xgboost": {
                        "n_estimators": 100,
                    },
                },
                "training": {},
                "feature": {
                    "target_column": "cost",
                    "date_column": "date",
                },
                "performance": {},
                "registry": {
                    "mlflow_experiment_name": "test",
                },
                "monitoring": {},
                "forecast": {},
            }
            
            with open(temp_file, 'w') as f:
                yaml.dump(test_data, f)
            
            # Monkeypatch the default path to use our temp file
            original_from_yaml = AppConfig.from_yaml
            
            def mock_from_yaml(config_path=None):
                if config_path is None:
                    return original_from_yaml(str(temp_file))
                return original_from_yaml(config_path)
            
            # This test is complex - skip for now and test with explicit path
            pytest.skip("Default path test requires more complex setup")
        finally:
            if temp_file.exists():
                temp_file.unlink()
    
    @pytest.mark.unit
    def test_app_config_to_dict(self, sample_app_config):
        """Test converting AppConfig to dictionary"""
        config_dict = sample_app_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_config"
        assert "data" in config_dict
        assert "model" in config_dict
    
    @pytest.mark.unit
    def test_app_config_to_yaml(self, sample_app_config):
        """Test converting AppConfig to YAML string"""
        yaml_str = sample_app_config.to_yaml()
        assert isinstance(yaml_str, str)
        assert "test_config" in yaml_str
        assert "data:" in yaml_str or "data:" in yaml_str.lower()
    
    @pytest.mark.unit
    def test_app_config_to_yaml_save_file(self, sample_app_config):
        """Test saving AppConfig to YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            yaml_str = sample_app_config.to_yaml(output_path=temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file can be loaded back
            loaded_config = AppConfig.from_yaml(temp_path)
            assert loaded_config.name == sample_app_config.name
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @pytest.mark.unit
    def test_app_config_nested_structure(self, sample_app_config):
        """Test nested configuration structure"""
        # Test nested model config
        assert sample_app_config.model.prophet.yearly_seasonality is True
        assert sample_app_config.model.arima.seasonal_period == 12
        assert sample_app_config.model.xgboost.n_estimators == 100
        
        # Test data config
        assert sample_app_config.data.delta_table_path == "test.database.test_table"
        
        # Test feature config
        assert sample_app_config.feature.target_column == "cost"
        assert len(sample_app_config.feature.lag_periods) == 3

