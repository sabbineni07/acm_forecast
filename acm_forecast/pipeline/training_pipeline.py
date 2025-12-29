"""
Training Pipeline
Section 7.1: Data Flow and Model Ingestion Diagram
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from pyspark.sql import SparkSession
import logging

from ..core import PluginFactory
from ..evaluation.model_evaluator import ModelEvaluator
from ..evaluation.model_comparison import ModelComparator
from ..config import AppConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline
    Section 7.1: Data Flow and Model Ingestion Diagram
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None):
        """
        Initialize training pipeline
        
        Args:
            config: AppConfig instance containing configuration
            spark: SparkSession for Databricks environment
        """
        self.config = config
        self.spark = spark
        
        # Create plugin factory
        self.factory = PluginFactory()
        
        # Initialize components using PluginFactory
        self.data_source = self.factory.create_data_source(config, spark, plugin_name="acm")
        self.data_prep = self.factory.create_data_preparation(config, spark, plugin_name="acm")
        self.data_quality = self.factory.create_data_quality(config, spark, plugin_name="default")
        self.feature_engineer = self.factory.create_feature_engineer(config, spark, plugin_name="default")
        self.model_registry = self.factory.create_model_registry(config, plugin_name="mlflow")
        
        # Models (created during training)
        self.prophet_forecaster = None
        self.arima_forecaster = None
        self.xgboost_forecaster = None
        
        # Evaluation
        self.evaluator = ModelEvaluator(config)
        self.comparator = ModelComparator(config)
    
    def run(self,
           category: str = "Total",
           start_date: Optional[str] = None,
           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            category: Cost category to train for
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training pipeline for {category}")
        
        # Step 1: Load data (Section 3.1)
        logger.info("Step 1: Loading data from Delta table")
        df_spark = self.data_source.load_data(
            start_date=start_date,
            end_date=end_date,
            category=category if category != "Total" else None
        )
        
        # Step 2: Data quality validation (Section 3.1.4)
        logger.info("Step 2: Validating data quality")
        quality_results = self.data_quality.comprehensive_validation(df_spark)
        
        if quality_results["quality_score"] < 80:
            logger.warning(f"Data quality score is {quality_results['quality_score']:.2f}%")
        
        # Step 3: Aggregate daily costs (Section 3.3.3)
        logger.info("Step 3: Aggregating daily costs")
        if category != "Total":
            daily_df_spark = self.data_prep.aggregate_data(
                df_spark, group_by=["meter_category"]
            )
        else:
            daily_df_spark = self.data_prep.aggregate_data(df_spark)
        
        # Convert to Pandas for modeling (after aggregation)
        date_col = self.config.feature.date_column
        
        # Step 4: Data preparation (Section 3.3)
        logger.info("Step 4: Preparing data")
        # Handle missing values using plugin method
        daily_df_spark = self.data_prep.handle_missing_values(daily_df_spark)
        daily_df = daily_df_spark.toPandas()
        daily_df = daily_df.sort_values(date_col).reset_index(drop=True)
        
        # Step 5: Split data (Section 3.3.2)
        logger.info("Step 5: Splitting data")
        train_df, val_df, test_df = self.data_prep.split(daily_df)
        
        # Step 6: Train models (Section 5.2.1)
        logger.info("Step 6: Training models")
        results = {}
        
        # Train Prophet
        try:
            logger.info("Training Prophet model")
            prophet_data = self.data_prep.prepare_for_training(train_df, model_type="prophet")
            self.prophet_forecaster = self.factory.create_model(config=self.config, category=category, plugin_name="prophet")
            self.prophet_forecaster.train(prophet_data)
            
            # Evaluate
            prophet_forecast = self.prophet_forecaster.predict(periods=len(test_df))
            prophet_metrics = self.prophet_forecaster.evaluate(
                prophet_forecast, self.data_prep.prepare_for_training(test_df, model_type="prophet")
            )
            results['prophet'] = {
                'model': self.prophet_forecaster,
                'metrics': prophet_metrics
            }
        except Exception as e:
            logger.error(f"Prophet training failed: {e}", exc_info=True)
            results['prophet'] = {'error': str(e)}
        
        # Train ARIMA
        try:
            logger.info("Training ARIMA model")
            arima_data = self.data_prep.prepare_for_training(train_df, model_type="arima")
            self.arima_forecaster = self.factory.create_model(config=self.config, category=category, plugin_name="arima")
            self.arima_forecaster.train(arima_data)
            
            # Evaluate
            arima_forecast_result = self.arima_forecaster.predict(periods=len(test_df), return_conf_int=True)
            if isinstance(arima_forecast_result, tuple):
                arima_forecast, _ = arima_forecast_result
            elif isinstance(arima_forecast_result, pd.DataFrame) and 'forecast' in arima_forecast_result.columns:
                arima_forecast = arima_forecast_result['forecast']
            else:
                arima_forecast = arima_forecast_result
            arima_metrics = self.arima_forecaster.evaluate(
                arima_forecast, self.data_prep.prepare_for_training(test_df, model_type="arima")
            )
            results['arima'] = {
                'model': self.arima_forecaster,
                'metrics': arima_metrics
            }
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}", exc_info=True)
            results['arima'] = {'error': str(e)}
        
        # Train XGBoost
        try:
            logger.info("Training XGBoost model")
            xgboost_data = self.feature_engineer.prepare_xgboost_features(train_df)
            self.xgboost_forecaster = self.factory.create_model(config=self.config, category=category, plugin_name="xgboost")
            self.xgboost_forecaster.train(xgboost_data)
            
            # Evaluate
            xgboost_forecast_result = self.xgboost_forecaster.predict(periods=0, df=self.feature_engineer.prepare_xgboost_features(test_df))
            if isinstance(xgboost_forecast_result, pd.DataFrame) and 'forecast' in xgboost_forecast_result.columns:
                xgboost_forecast = xgboost_forecast_result['forecast'].values
            else:
                xgboost_forecast = xgboost_forecast_result
            xgboost_metrics = self.xgboost_forecaster.evaluate(
                xgboost_forecast, test_df[self.config.feature.target_column].values
            )
            results['xgboost'] = {
                'model': self.xgboost_forecaster,
                'metrics': xgboost_metrics
            }
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}", exc_info=True)
            results['xgboost'] = {'error': str(e)}
        
        # Step 7: Compare models (Section 6.1)
        logger.info("Step 7: Comparing models")
        # ... comparison logic ...
        
        # Step 8: Register best model (Section 7.2)
        logger.info("Step 8: Registering models")
        # ... registration logic ...
        
        logger.info(f"Training pipeline completed for {category}")
        return results
