"""
Default Feature Engineering Plugin

PRIMARY IMPLEMENTATION for feature engineering.
The actual implementation is here - FeatureEngineer class delegates to this.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import logging

from ...core.interfaces import IFeatureEngineer
from ...core.base_plugin import BasePlugin
from ...config import AppConfig

logger = logging.getLogger(__name__)


class DefaultFeatureEngineer(BasePlugin, IFeatureEngineer):
    """
    Default feature engineering plugin - PRIMARY IMPLEMENTATION
    Section 3.3.4: Variable Creation
    Section 5.1: Variable Selection and Feature Engineering
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None, **kwargs):
        """Initialize default feature engineering plugin"""
        super().__init__(config, spark, **kwargs)
    
    def create_temporal_features(self, df: pd.DataFrame,
                                date_col: str = None) -> pd.DataFrame:
        """
        Create time-based features (Section 3.3.4)
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with temporal features
        """
        if date_col is None:
            date_col = self.config.feature.date_column
        df_features = df.copy()
        df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        # Basic time features
        df_features['Year'] = df_features[date_col].dt.year
        df_features['Month'] = df_features[date_col].dt.month
        df_features['Day'] = df_features[date_col].dt.day
        df_features['DayOfWeek'] = df_features[date_col].dt.dayofweek
        df_features['DayOfYear'] = df_features[date_col].dt.dayofyear
        df_features['WeekOfYear'] = df_features[date_col].dt.isocalendar().week
        df_features['Quarter'] = df_features[date_col].dt.quarter
        
        # Boolean features
        df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
        df_features['IsMonthStart'] = df_features[date_col].dt.is_month_start.astype(int)
        df_features['IsMonthEnd'] = df_features[date_col].dt.is_month_end.astype(int)
        df_features['IsQuarterStart'] = df_features[date_col].dt.is_quarter_start.astype(int)
        df_features['IsQuarterEnd'] = df_features[date_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding
        df_features['Month_sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
        df_features['Month_cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
        df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
        df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
        df_features['DayOfYear_sin'] = np.sin(2 * np.pi * df_features['DayOfYear'] / 365)
        df_features['DayOfYear_cos'] = np.cos(2 * np.pi * df_features['DayOfYear'] / 365)
        
        logger.info("Created temporal features")
        return df_features
    
    def create_lag_features(self, 
                           df: pd.DataFrame,
                           target_col: str = None,
                           group_col: Optional[str] = None,
                           lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lag features (Section 3.3.4)
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            group_col: Grouping column (e.g., 'MeterCategory')
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        if target_col is None:
            target_col = self.config.feature.target_column
        if lags is None:
            lags = self.config.feature.lag_features or [1, 2, 3, 7, 14, 30]
        
        df_lags = df.copy()
        date_col = self.config.feature.date_column
        df_lags = df_lags.sort_values(date_col)
        
        if group_col:
            # Create lags by group
            for lag in lags:
                df_lags[f'{target_col}_lag_{lag}'] = (
                    df_lags.groupby(group_col)[target_col].shift(lag)
                )
        else:
            # Create lags for entire series
            for lag in lags:
                df_lags[f'{target_col}_lag_{lag}'] = df_lags[target_col].shift(lag)
        
        logger.info(f"Created lag features: {lags}")
        return df_lags
    
    def prepare_features(self, df: pd.DataFrame, model_type: str = "xgboost") -> pd.DataFrame:
        """
        Prepare features for model training
        
        Args:
            df: Input DataFrame
            model_type: Type of model
            
        Returns:
            DataFrame with prepared features
        """
        if model_type == "xgboost":
            return self.prepare_xgboost_features(df)
        else:
            return self.create_temporal_features(df)
    
    # Additional methods for backward compatibility
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               target_col: str = None,
                               group_col: Optional[str] = None,
                               windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create rolling window features (Section 3.3.4)
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            group_col: Grouping column
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        if target_col is None:
            target_col = self.config.feature.target_column
        if windows is None:
            windows = self.config.feature.rolling_features.get('window_sizes', [3, 7, 14, 30]) if hasattr(self.config.feature, 'rolling_features') else [3, 7, 14, 30]
        
        df_rolling = df.copy()
        date_col = self.config.feature.date_column
        df_rolling = df_rolling.sort_values(date_col)
        
        if group_col:
            # Create rolling features by group
            for window in windows:
                df_rolling[f'{target_col}_rolling_mean_{window}'] = (
                    df_rolling.groupby(group_col)[target_col]
                    .rolling(window=window)
                    .mean()
                    .reset_index(0, drop=True)
                )
                df_rolling[f'{target_col}_rolling_std_{window}'] = (
                    df_rolling.groupby(group_col)[target_col]
                    .rolling(window=window)
                    .std()
                    .reset_index(0, drop=True)
                )
                df_rolling[f'{target_col}_rolling_max_{window}'] = (
                    df_rolling.groupby(group_col)[target_col]
                    .rolling(window=window)
                    .max()
                    .reset_index(0, drop=True)
                )
                df_rolling[f'{target_col}_rolling_min_{window}'] = (
                    df_rolling.groupby(group_col)[target_col]
                    .rolling(window=window)
                    .min()
                    .reset_index(0, drop=True)
                )
        else:
            # Create rolling features for entire series
            for window in windows:
                df_rolling[f'{target_col}_rolling_mean_{window}'] = (
                    df_rolling[target_col].rolling(window=window).mean()
                )
                df_rolling[f'{target_col}_rolling_std_{window}'] = (
                    df_rolling[target_col].rolling(window=window).std()
                )
                df_rolling[f'{target_col}_rolling_max_{window}'] = (
                    df_rolling[target_col].rolling(window=window).max()
                )
                df_rolling[f'{target_col}_rolling_min_{window}'] = (
                    df_rolling[target_col].rolling(window=window).min()
                )
        
        logger.info(f"Created rolling features: {windows}")
        return df_rolling
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features (Section 3.3.4)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features
        """
        df_derived = df.copy()
        target_col = self.config.feature.target_column
        
        # Cost per unit
        if 'quantity' in df_derived.columns and target_col in df_derived.columns:
            df_derived['CostPerUnit'] = (
                df_derived[target_col] / (df_derived['quantity'] + 1e-8)
            )
        
        # Growth rates (if lag features exist)
        lag_1_col = f'{target_col}_lag_1'
        if lag_1_col in df_derived.columns:
            df_derived['DayOverDayChange'] = (
                df_derived[target_col] - df_derived[lag_1_col]
            )
            df_derived['DayOverDayPctChange'] = (
                (df_derived[target_col] - df_derived[lag_1_col]) /
                (df_derived[lag_1_col] + 1e-8) * 100
            )
        
        lag_7_col = f'{target_col}_lag_7'
        if lag_7_col in df_derived.columns:
            df_derived['WeekOverWeekChange'] = (
                df_derived[target_col] - df_derived[lag_7_col]
            )
        
        lag_30_col = f'{target_col}_lag_30'
        if lag_30_col in df_derived.columns:
            df_derived['MonthOverMonthChange'] = (
                df_derived[target_col] - df_derived[lag_30_col]
            )
        
        logger.info("Created derived features")
        return df_derived
    
    def prepare_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare all features for XGBoost (Section 5.1)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all XGBoost features
        """
        # Create all feature types
        df_features = self.create_temporal_features(df)
        df_features = self.create_lag_features(df_features)
        df_features = self.create_rolling_features(df_features)
        df_features = self.create_derived_features(df_features)
        
        logger.info("Prepared XGBoost features")
        return df_features
