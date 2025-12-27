"""
XGBoost Model Plugin

PRIMARY IMPLEMENTATION for XGBoost time series forecasting.
The actual implementation is here - XGBoostForecaster class delegates to this.
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from ...core.interfaces import IModel
from ...core.base_plugin import BasePlugin
from ...config import AppConfig
from ...data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class XGBoostModelPlugin(BasePlugin, IModel):
    """
    XGBoost model plugin - PRIMARY IMPLEMENTATION
    Section 4.2.1: XGBoost Model Methodology
    Section 5.2.1: Model Estimation - XGBoost
    Section 5.2.3: Final Model Specification - XGBoost
    Section 5.2.4: Model Diagnostics - XGBoost
    """
    
    def __init__(self, config: AppConfig, category: str = "Total", **kwargs):
        """Initialize XGBoost model plugin
        
        Args:
            config: AppConfig instance
            category: Cost category name
            **kwargs: Plugin-specific configuration
        """
        super().__init__(config, None, **kwargs)  # XGBoost doesn't need Spark
        self.category = category
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(config)
        self.is_trained = False
        self.feature_names = None
    
    def create_model(self) -> xgb.XGBRegressor:
        """
        Create XGBoost model with configuration (Section 5.2.3)
        
        Returns:
            Configured XGBoost model
        """
        xgboost_config = self.config.model.xgboost
        model = xgb.XGBRegressor(
            n_estimators=xgboost_config.n_estimators or 100,
            max_depth=xgboost_config.max_depth or 6,
            learning_rate=xgboost_config.learning_rate or 0.1,
            subsample=xgboost_config.subsample or 0.8,
            colsample_bytree=xgboost_config.colsample_bytree or 0.8,
            objective=xgboost_config.objective or "reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Created XGBoost model for {self.category}")
        return model
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train XGBoost model (Section 5.2.1)
        
        Args:
            df: Training DataFrame with features
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training XGBoost model for {self.category}")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Time series split (chronological)
        test_size = 0.2
        validation_size = 0.1
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
        xgboost_config = self.config.model.xgboost
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=xgboost_config.early_stopping_rounds or 10,
            verbose=False
        )
        
        self.is_trained = True
        
        logger.info(f"XGBoost model trained for {self.category}")
        return {
            "model": self.model,
            "is_trained": True,
            "category": self.category,
            "feature_names": self.feature_names
        }
    
    def predict(self, periods: int = 30, **kwargs) -> pd.DataFrame:
        """
        Generate predictions (Section 5.2.1)
        
        Note: XGBoost predict requires a DataFrame with features.
        The periods parameter is ignored for XGBoost, use df parameter instead.
        
        Args:
            periods: Not used for XGBoost (maintained for interface compatibility)
            **kwargs: Must include 'df' parameter with DataFrame containing features
            
        Returns:
            DataFrame with forecast values
        """
        if 'df' not in kwargs:
            raise ValueError("XGBoost predict requires 'df' parameter with DataFrame containing features")
        
        df = kwargs['df']
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
        return pd.DataFrame({'forecast': forecast})
    
    def save(self, path: str) -> None:
        """
        Save model to path
        
        Args:
            path: Path to save model
        """
        if self.model is not None:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }, f)
            logger.info(f"Saved XGBoost model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from path
        
        Args:
            path: Path to load model from
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        self.is_trained = True
        logger.info(f"Loaded XGBoost model from {path}")
    
    # Additional methods for backward compatibility
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
        exclude_cols = [self.config.feature.target_column, self.config.feature.date_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features (one-hot encoding) - using snake_case
        categorical_cols = ['meter_category', 'resource_location', 'plan_name']
        for col in categorical_cols:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
                # Update feature columns
                feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df[self.config.feature.target_column].copy()
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Handle missing values in features
        X = X.fillna(0)
        
        logger.info(f"Prepared {len(feature_cols)} features for {self.category}")
        return X, y
    
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
