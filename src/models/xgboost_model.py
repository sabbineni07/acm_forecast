"""
XGBoost Model Implementation
Section 4.2.1: Model Methodology - XGBoost
Section 5.2.1: Model Estimation - XGBoost
Section 5.2.3: Final Model Specification - XGBoost
Section 5.2.4: Model Diagnostics - XGBoost
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ..config.settings import model_config, feature_config
from ..data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """
    XGBoost model for time series forecasting
    Section 4.2.1: XGBoost Model Methodology
    """
    
    def __init__(self, category: str = "Total"):
        """
        Initialize XGBoost forecaster
        
        Args:
            category: Cost category name
        """
        self.category = category
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.feature_names = None
    
    def create_model(self) -> xgb.XGBRegressor:
        """
        Create XGBoost model with configuration (Section 5.2.3)
        
        Returns:
            Configured XGBoost model
        """
        model = xgb.XGBRegressor(
            n_estimators=model_config.xgboost_n_estimators,
            max_depth=model_config.xgboost_max_depth,
            learning_rate=model_config.xgboost_learning_rate,
            subsample=model_config.xgboost_subsample,
            colsample_bytree=model_config.xgboost_colsample_bytree,
            objective=model_config.xgboost_objective,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Created XGBoost model for {self.category}")
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for XGBoost (Section 5.1)
        
        Args:
            df: Input DataFrame with temporal and lag features
            
        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        # Ensure features are created
        if 'Month' not in df.columns:
            df = self.feature_engineer.prepare_xgboost_features(df)
        
        # Select feature columns (exclude target and date)
        exclude_cols = [feature_config.target_column, feature_config.date_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features (one-hot encoding)
        categorical_cols = ['MeterCategory', 'ResourceLocation', 'ServiceTier']
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
                # Update feature columns
                feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df[feature_config.target_column].copy()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing values in features
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(feature_cols)} features for {self.category}")
        return X, y
    
    def train(self, 
              df: pd.DataFrame,
              test_size: float = 0.2,
              validation_size: float = 0.1) -> xgb.XGBRegressor:
        """
        Train XGBoost model (Section 5.2.1)
        
        Args:
            df: Training DataFrame with features
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            
        Returns:
            Trained XGBoost model
        """
        logger.info(f"Training XGBoost model for {self.category}")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Time series split (chronological)
        # For time series, we should split chronologically, not randomly
        split_idx = int(len(X) * (1 - test_size - validation_size))
        val_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:val_idx]
        X_test = X.iloc[val_idx:]
        
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:val_idx]
        y_test = y.iloc[val_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        if self.model is None:
            self.model = self.create_model()
        
        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=model_config.xgboost_early_stopping_rounds,
            verbose=False
        )
        
        self.is_trained = True
        
        logger.info(f"XGBoost model trained for {self.category}")
        return self.model
    
    def predict(self, 
                df: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts (Section 5.2.1)
        
        Args:
            df: DataFrame with features (must have same features as training)
            
        Returns:
            Forecast array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        # Ensure same features as training
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)
        
        if missing_features:
            for feat in missing_features:
                X[feat] = 0
        
        if extra_features:
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        forecast = self.model.predict(X_scaled)
        
        logger.info(f"Generated forecast for {self.category}")
        return forecast
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (Section 5.2.4)
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance calculated for {self.category}")
        return importance
    
    def evaluate(self,
                 forecast: np.ndarray,
                 actual: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            forecast: Forecast values
            actual: Actual values
            
        Returns:
            Dictionary of evaluation metrics
        """
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100
        r2 = r2_score(actual, forecast)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
        
        logger.info(f"Evaluation metrics for {self.category}: {metrics}")
        return metrics


