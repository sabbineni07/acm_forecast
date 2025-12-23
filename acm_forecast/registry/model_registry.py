"""
Model Registry Module
Section 7.2: Model Registry Configuration in Databricks
"""

from typing import Dict, Optional, Any
import mlflow
from mlflow.tracking import MlflowClient
import logging

from ..config import AppConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    MLflow Model Registry integration
    Section 7.2: Model Registry Configuration in Databricks
    """
    
    def __init__(self, config: AppConfig, tracking_uri: Optional[str] = None):
        """
        Initialize model registry
        
        Args:
            config: AppConfig instance containing configuration
            tracking_uri: MLflow tracking URI (optional override)
        """
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
            logger.warning(f"Could not set experiment: {e}")
    
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
                mlflow.log_artifact(joblib.dump(model, "model.pkl")[0], "model")
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            else:
                import joblib
                mlflow.log_artifact(joblib.dump(model, "model.pkl")[0], "model")
            
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
    
    def load_model(self,
                   model_name: str,
                   stage: str = "Production") -> Any:
        """
        Load model from registry (Section 7.2)
        
        Args:
            model_name: Name of the model
            stage: Model stage ('Staging' or 'Production')
            
        Returns:
            Loaded model object
        """
        model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{stage}"
        )
        
        logger.info(f"Loaded {model_name} from {stage} stage")
        return model
    
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


