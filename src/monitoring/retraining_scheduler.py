"""
Retraining Scheduler Module
Section 8.3: Model Retraining Schedule
"""

from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import logging

from ..config import AppConfig
from ..monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Schedule and trigger model retraining
    Section 8.3: Model Retraining Schedule
    """
    
    def __init__(self, config: AppConfig, performance_monitor: Optional[PerformanceMonitor] = None):
        """
        Initialize retraining scheduler
        
        Args:
            config: AppConfig instance containing configuration
            performance_monitor: PerformanceMonitor instance
        """
        self.config = config
        self.performance_monitor = performance_monitor or PerformanceMonitor(config)
        self.last_retraining_date = {}
        self.retraining_history = []
    
    def should_retrain(self,
                      model_name: str,
                      current_mape: Optional[float] = None,
                      last_retraining: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Determine if model should be retrained (Section 8.3)
        
        Args:
            model_name: Name of the model
            current_mape: Current MAPE (if available)
            last_retraining: Last retraining date (if available)
            
        Returns:
            Dictionary with retraining decision
        """
        triggers = []
        should_retrain = False
        
        # Check performance degradation
        trigger_mape = self.config.monitoring.retraining_trigger_mape or 15.0
        if current_mape and current_mape > trigger_mape:
            triggers.append("performance_degradation")
            should_retrain = True
        
        # Check time since last retraining
        if last_retraining:
            months_since_retraining = (
                (datetime.now() - last_retraining).days / 30
            )
            
            max_months = self.config.monitoring.max_months_without_retraining or 6
            if months_since_retraining >= max_months:
                triggers.append("time_based")
                should_retrain = True
            
            # Monthly retraining
            monthly_retraining = self.config.monitoring.monthly_retraining
            if monthly_retraining and months_since_retraining >= 1:
                triggers.append("monthly_schedule")
                should_retrain = True
            
            # Quarterly retraining
            quarterly_retraining = self.config.monitoring.quarterly_retraining
            if quarterly_retraining and months_since_retraining >= 3:
                triggers.append("quarterly_schedule")
                should_retrain = True
        else:
            # No previous retraining
            triggers.append("initial_training")
            should_retrain = True
        
        result = {
            "model_name": model_name,
            "should_retrain": should_retrain,
            "triggers": triggers,
            "current_mape": current_mape,
            "last_retraining": last_retraining,
            "timestamp": datetime.now()
        }
        
        if should_retrain:
            logger.info(f"Retraining recommended for {model_name}: {triggers}")
        
        return result
    
    def schedule_retraining(self,
                           model_name: str,
                           retraining_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Schedule model retraining
        
        Args:
            model_name: Name of the model
            retraining_date: Date to retrain (default: now)
            
        Returns:
            Retraining schedule information
        """
        if retraining_date is None:
            retraining_date = datetime.now()
        
        schedule = {
            "model_name": model_name,
            "scheduled_date": retraining_date,
            "status": "scheduled",
            "created_at": datetime.now()
        }
        
        self.retraining_history.append(schedule)
        
        logger.info(f"Scheduled retraining for {model_name} on {retraining_date}")
        return schedule
    
    def record_retraining(self,
                         model_name: str,
                         retraining_date: Optional[datetime] = None) -> None:
        """
        Record that retraining was completed
        
        Args:
            model_name: Name of the model
            retraining_date: Date retraining was completed (default: now)
        """
        if retraining_date is None:
            retraining_date = datetime.now()
        
        self.last_retraining_date[model_name] = retraining_date
        
        logger.info(f"Recorded retraining for {model_name} on {retraining_date}")


