"""
MLflow Model Registry Plugin

PRIMARY IMPLEMENTATION for MLflow model registry.
The actual implementation is here - ModelRegistry class delegates to this.
"""

from typing import Dict, Optional, Any
import mlflow
from mlflow.tracking import MlflowClient
import logging

from ...core.interfaces import IModelRegistry
from ...core.base_plugin import BasePlugin
from ...config import AppConfig

logger = logging.getLogger(__name__)


class MLflowModelRegistry(BasePlugin, IModelRegistry):
    """
    MLflow Model Registry plugin - PRIMARY IMPLEMENTATION
    Section 7.2: Model Registry Configuration in Databricks
    """
    
    def __init__(self, config: AppConfig, tracking_uri: Optional[str] = None, **kwargs):
        """Initialize MLflow model registry plugin
        
        Args:
            config: AppConfig instance
            tracking_uri: MLflow tracking URI (optional override)
            **kwargs: Plugin-specific configuration
        """
        super().__init__(config, None, **kwargs)  # ModelRegistry doesn't need Spark
        self.config = config
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif config.registry.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.registry.mlflow_tracking_uri)
        
        self.client = MlflowClient()
        self.experiment_name = config.registry.mlflow_experiment_name
        
        # Set experiment
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}", exc_info=True)
    
    def save_model(self, model: Any, name: str, version: Optional[str] = None) -> str:
        """
        Save model to registry
        
        Args:
            model: Model object to save
            name: Model name
            version: Optional version string (ignored, MLflow handles versioning)
            
        Returns:
            Model version string
        """
        # MLflow auto-versions, so version parameter is ignored
        # This method signature matches the interface but MLflow handles versioning differently
        # Use register_model for full registration instead
        raise NotImplementedError("Use register_model() for MLflow registration")
    
    def load_model(self, name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Any:
        """
        Load model from registry
        
        Args:
            name: Model name
            version: Optional version string
            stage: Optional stage string (defaults to 'Production' if version not provided)
            
        Returns:
            Loaded model object
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        else:
            # Load from specified stage (default Production)
            stage = stage or "Production"
            model_uri = f"models:/{name}/{stage}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded {name} from registry")
        return model
    
    # Additional methods for backward compatibility (not in interface but used by original class)
    def register_model(self,
                      model: Any,
                      model_name: str,
                      model_type: str,
                      metrics: Dict[str, float],
                      category: str = "Total",
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register model in MLflow (Section 7.2)
        
        Args:
            model: Trained model object
            model_name: Name for the model in registry
            model_type: Type of model ('prophet', 'arima', 'xgboost')
            metrics: Performance metrics dictionary
            category: Cost category
            tags: Additional tags
            
        Returns:
            Model version string
        """
        with mlflow.start_run():
            # Log model
            if model_type == 'prophet':
                mlflow.prophet.log_model(model, "model")
            elif model_type == 'arima':
                # ARIMA models need to be saved differently
                import joblib
                temp_path = "model.pkl"
                joblib.dump(model, temp_path)
                mlflow.log_artifact(temp_path, "model")
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            else:
                import joblib
                temp_path = "model.pkl"
                joblib.dump(model, temp_path)
                mlflow.log_artifact(temp_path, "model")
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            mlflow.log_param("category", category)
            mlflow.log_param("model_type", model_type)
            
            # Log tags
            if tags:
                for tag_name, tag_value in tags.items():
                    mlflow.set_tag(tag_name, tag_value)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered {model_name} version {model_version.version}")
            return model_version.version
    
    def promote_model(self,
                     model_name: str,
                     version: str,
                     stage: str = "Staging") -> None:
        """
        Promote model to stage (Section 7.2)
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage ('Staging' or 'Production')
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Promoted {model_name} version {version} to {stage}")
    
    def get_model_versions(self, model_name: str) -> list:
        """
        Get all versions of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")
        return versions
    
    def get_latest_version(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """
        Get latest version of model in stage
        
        Args:
            model_name: Name of the model
            stage: Model stage
            
        Returns:
            Latest version string or None
        """
        versions = self.get_model_versions(model_name)
        
        # Filter by stage and get latest
        stage_versions = [v for v in versions if v.current_stage == stage]
        if stage_versions:
            latest = max(stage_versions, key=lambda x: int(x.version))
            return latest.version
        
        return None
