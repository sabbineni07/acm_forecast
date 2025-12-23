"""
Model Versioning Module
Section 7.3: Access, Versioning and Controls Description
"""

from typing import Dict, Optional, Any
import logging
from datetime import datetime

from .model_registry import ModelRegistry
from ..config import AppConfig

logger = logging.getLogger(__name__)


class ModelVersioning:
    """
    Model versioning and control
    Section 7.3: Access, Versioning and Controls Description
    """
    
    def __init__(self, config: AppConfig, registry: Optional[ModelRegistry] = None):
        """
        Initialize model versioning
        
        Args:
            config: AppConfig instance containing configuration
            registry: ModelRegistry instance
        """
        self.config = config
        self.registry = registry or ModelRegistry(config)
    
    def create_version_tag(self,
                          category: str,
                          training_date: str,
                          data_version: str,
                          performance_metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Create version tags (Section 7.3)
        
        Args:
            category: Cost category
            training_date: Date model was trained
            data_version: Version of training data
            performance_metrics: Performance metrics
            
        Returns:
            Dictionary of tags
        """
        tags = {
            "category": category,
            "training_date": training_date,
            "data_version": data_version,
            "mape": str(performance_metrics.get("mape", "N/A")),
            "r2": str(performance_metrics.get("r2", "N/A")),
            "created_at": datetime.now().isoformat()
        }
        
        return tags
    
    def compare_versions(self,
                       model_name: str,
                       version1: str,
                       version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        # Get version information
        versions = self.registry.get_model_versions(model_name)
        v1_info = next((v for v in versions if v.version == version1), None)
        v2_info = next((v for v in versions if v.version == version2), None)
        
        if not v1_info or not v2_info:
            raise ValueError("One or both versions not found")
        
        comparison = {
            "version1": {
                "version": version1,
                "stage": v1_info.current_stage,
                "created_at": v1_info.creation_timestamp
            },
            "version2": {
                "version": version2,
                "stage": v2_info.current_stage,
                "created_at": v2_info.creation_timestamp
            }
        }
        
        logger.info(f"Compared versions {version1} and {version2} of {model_name}")
        return comparison
    
    def rollback_model(self,
                      model_name: str,
                      target_version: str) -> None:
        """
        Rollback to previous version (Section 7.3)
        
        Args:
            model_name: Name of the model
            target_version: Version to rollback to
        """
        # Get current production version
        current_version = self.registry.get_latest_version(model_name, "Production")
        
        if current_version:
            # Transition current to Archived
            self.registry.client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )
        
        # Promote target version to Production
        self.registry.promote_model(model_name, target_version, "Production")
        
        logger.info(f"Rolled back {model_name} to version {target_version}")


