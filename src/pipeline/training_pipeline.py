"""
Training Pipeline
Section 7.1: Data Flow and Model Ingestion Diagram
"""

from typing import Dict, List, Optional
import pandas as pd
from pyspark.sql import SparkSession
import logging

from ..data.data_source import DataSource
from ..data.data_preparation import DataPreparation
from ..data.data_quality import DataQualityValidator
from ..data.feature_engineering import FeatureEngineer
from ..models.prophet_model import ProphetForecaster
from ..models.arima_model import ARIMAForecaster
from ..models.xgboost_model import XGBoostForecaster
from ..evaluation.model_evaluator import ModelEvaluator
from ..evaluation.model_comparison import ModelComparator
from ..registry.model_registry import ModelRegistry
from ..config.settings import data_config, feature_config

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline
    Section 7.1: Data Flow and Model Ingestion Diagram
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize training pipeline
        
        Args:
            spark: SparkSession for Databricks environment
        """
        self.spark = spark
        
        # Initialize components
        self.data_source = DataSource(spark)
        self.data_prep = DataPreparation(spark)
        self.data_quality = DataQualityValidator(spark)
        self.feature_engineer = FeatureEngineer(spark)
        self.model_registry = ModelRegistry()
        
        # Models
        self.prophet_forecaster = None
        self.arima_forecaster = None
        self.xgboost_forecaster = None
        
        # Evaluation
        self.evaluator = ModelEvaluator()
        self.comparator = ModelComparator()
    
    def run(self,
           category: str = "Total",
           start_date: Optional[str] = None,
           end_date: Optional[str] = None) -> Dict[str, any]:
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
        df_spark = self.data_source.load_from_delta(
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
            daily_df_spark = self.data_prep.aggregate_daily_costs(
                df_spark, group_by=["MeterCategory"]
            )
        else:
            daily_df_spark = self.data_prep.aggregate_daily_costs(df_spark)
        
        # Convert to Pandas for modeling
        daily_df = daily_df_spark.toPandas()
        daily_df = daily_df.sort_values("UsageDateTime").reset_index(drop=True)
        
        # Step 4: Data preparation (Section 3.3)
        logger.info("Step 4: Preparing data")
        daily_df = self.data_prep.handle_missing_values(
            self.spark.createDataFrame(daily_df) if self.spark else pd.DataFrame()
        )
        if self.spark:
            daily_df = daily_df.toPandas()
        
        # Step 5: Split data (Section 3.3.2)
        logger.info("Step 5: Splitting data")
        train_df, val_df, test_df = self.data_prep.split_time_series(daily_df)
        
        # Step 6: Train models (Section 5.2.1)
        logger.info("Step 6: Training models")
        results = {}
        
        # Train Prophet
        try:
            logger.info("Training Prophet model")
            prophet_data = self.data_prep.prepare_for_prophet(train_df)
            self.prophet_forecaster = ProphetForecaster(category)
            self.prophet_forecaster.train(prophet_data)
            
            # Evaluate
            prophet_forecast = self.prophet_forecaster.predict(periods=len(test_df))
            prophet_metrics = self.prophet_forecaster.evaluate(
                prophet_forecast, self.data_prep.prepare_for_prophet(test_df)
            )
            results['prophet'] = {
                'model': self.prophet_forecaster,
                'metrics': prophet_metrics
            }
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            results['prophet'] = {'error': str(e)}
        
        # Train ARIMA
        try:
            logger.info("Training ARIMA model")
            arima_data = self.data_prep.prepare_for_arima(train_df)
            self.arima_forecaster = ARIMAForecaster(category)
            self.arima_forecaster.train(arima_data)
            
            # Evaluate
            arima_forecast, _ = self.arima_forecaster.predict(n_periods=len(test_df))
            arima_metrics = self.arima_forecaster.evaluate(
                arima_forecast, self.data_prep.prepare_for_arima(test_df)
            )
            results['arima'] = {
                'model': self.arima_forecaster,
                'metrics': arima_metrics
            }
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            results['arima'] = {'error': str(e)}
        
        # Train XGBoost
        try:
            logger.info("Training XGBoost model")
            xgboost_data = self.feature_engineer.prepare_xgboost_features(train_df)
            self.xgboost_forecaster = XGBoostForecaster(category)
            self.xgboost_forecaster.train(xgboost_data)
            
            # Evaluate
            xgboost_forecast = self.xgboost_forecaster.predict(
                self.feature_engineer.prepare_xgboost_features(test_df)
            )
            xgboost_metrics = self.xgboost_forecaster.evaluate(
                xgboost_forecast, test_df[feature_config.target_column].values
            )
            results['xgboost'] = {
                'model': self.xgboost_forecaster,
                'metrics': xgboost_metrics
            }
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results['xgboost'] = {'error': str(e)}
        
        # Step 7: Compare models (Section 6.1)
        logger.info("Step 7: Comparing models")
        # ... comparison logic ...
        
        # Step 8: Register best model (Section 7.2)
        logger.info("Step 8: Registering models")
        # ... registration logic ...
        
        logger.info(f"Training pipeline completed for {category}")
        return results

