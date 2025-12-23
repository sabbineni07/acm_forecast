"""
Feature Engineering Module
Section 3.3.4: Variable Creation
Section 5.1: Variable Selection and Feature Engineering
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, year, month, dayofmonth, dayofweek, dayofyear,
    weekofyear, quarter, sin, cos, lit, when
)
import logging

from ..config import AppConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for time series forecasting
    Section 3.3.4: Variable Creation
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None):
        """
        Initialize feature engineer
        
        Args:
            config: AppConfig instance containing configuration
            spark: SparkSession for Databricks environment
        """
        self.config = config
        self.spark = spark
    
    def create_temporal_features(self, df: pd.DataFrame,
                                date_col: str = "UsageDateTime") -> pd.DataFrame:
        """
        Create time-based features (Section 3.3.4)
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with temporal features
        """
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
                           target_col: str = "PreTaxCost",
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
        if lags is None:
            lags = self.config.feature.lag_periods or [1, 2, 3, 7, 14, 30]
        
        df_lags = df.copy()
        df_lags = df_lags.sort_values('UsageDateTime')
        
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
    
    def create_rolling_features(self,
                               df: pd.DataFrame,
                               target_col: str = "PreTaxCost",
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
        if windows is None:
            windows = self.config.feature.rolling_windows or [3, 7, 14, 30]
        
        df_rolling = df.copy()
        df_rolling = df_rolling.sort_values('UsageDateTime')
        
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
        
        # Cost per unit
        if 'UsageQuantity' in df_derived.columns and 'PreTaxCost' in df_derived.columns:
            df_derived['CostPerUnit'] = (
                df_derived['PreTaxCost'] / (df_derived['UsageQuantity'] + 1e-8)
            )
        
        # Growth rates (if lag features exist)
        if 'PreTaxCost_lag_1' in df_derived.columns:
            df_derived['DayOverDayChange'] = (
                df_derived['PreTaxCost'] - df_derived['PreTaxCost_lag_1']
            )
            df_derived['DayOverDayPctChange'] = (
                (df_derived['PreTaxCost'] - df_derived['PreTaxCost_lag_1']) /
                (df_derived['PreTaxCost_lag_1'] + 1e-8) * 100
            )
        
        if 'PreTaxCost_lag_7' in df_derived.columns:
            df_derived['WeekOverWeekChange'] = (
                df_derived['PreTaxCost'] - df_derived['PreTaxCost_lag_7']
            )
        
        if 'PreTaxCost_lag_30' in df_derived.columns:
            df_derived['MonthOverMonthChange'] = (
                df_derived['PreTaxCost'] - df_derived['PreTaxCost_lag_30']
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


