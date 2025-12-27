"""
Default Forecaster Plugin

Wraps forecast pipeline functionality to implement IForecaster interface.
"""

from typing import Dict, Any, Optional

from ...core.interfaces import IForecaster
from ...core.base_plugin import BasePlugin
from ...config import AppConfig
from ...pipeline.forecast_pipeline import ForecastPipeline as OriginalForecastPipeline


class DefaultForecaster(BasePlugin, IForecaster):
    """Default forecaster plugin implementation"""
    
    def __init__(self, config: AppConfig, **kwargs):
        """Initialize default forecaster plugin"""
        super().__init__(config, None, **kwargs)
        self._impl = OriginalForecastPipeline(config)
    
    def generate_forecast(self, 
                         model_name: str,
                         horizon_days: int,
                         **kwargs) -> Dict[str, Any]:
        """Generate forecast"""
        category = kwargs.get('category', 'Total')
        horizons = kwargs.get('horizons', [horizon_days])
        
        return self._impl.generate_forecasts(
            category=category,
            horizons=horizons,
            model_name=model_name
        )

