"""
Data Quality Validation Module
Section 3.1.4: Data Reliability
"""

from typing import Dict, Any, Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, isnull, isnan
import logging

from ..config import AppConfig

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Data quality validation and monitoring
    Section 3.1.4: Data Reliability
    """
    
    def __init__(self, config: AppConfig, spark: Optional[SparkSession] = None):
        """
        Initialize data quality validator
        
        Args:
            config: AppConfig instance containing configuration
            spark: SparkSession for Databricks environment
        """
        self.config = config
        self.spark = spark
    
    def validate_completeness(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data completeness (Section 3.1.4)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Completeness validation results
        """
        total_records = df.count()
        
        # Check for missing values in key columns (using snake_case)
        key_columns = [
            self.config.feature.date_column,  # usage_date
            self.config.feature.target_column,  # cost_in_billing_currency
            "meter_category",
            "resource_location",
            "billing_currency_code"
        ]
        
        missing_counts = {}
        for col_name in key_columns:
            if col_name in df.columns:
                missing_count = df.filter(
                    isnull(col(col_name)) | isnan(col(col_name))
                ).count()
                missing_counts[col_name] = {
                    "count": missing_count,
                    "percentage": (missing_count / total_records * 100) if total_records > 0 else 0
                }
        
        # Overall completeness
        total_missing = sum([m["count"] for m in missing_counts.values()])
        completeness_rate = ((total_records - total_missing) / total_records * 100) if total_records > 0 else 0
        
        threshold = self.config.performance.warning_missing_data or 5.0
        results = {
            "total_records": total_records,
            "missing_values": missing_counts,
            "completeness_rate": completeness_rate,
            "meets_threshold": completeness_rate >= (100 - threshold)
        }
        
        return results
    
    def validate_accuracy(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data accuracy (Section 3.1.4)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Accuracy validation results
        """
        # Check for negative costs (should not exist)
        target_col = self.config.feature.target_column
        negative_costs = df.filter(col(target_col) < 0).count()
        
        # Check for zero costs (may be valid, but flag for review)
        zero_costs = df.filter(col(target_col) == 0).count()
        
        # Check currency consistency (should be USD)
        currency_check = df.filter(col("billing_currency_code") != "USD").count() if "billing_currency_code" in df.columns else 0
        
        # Check date range validity
        date_col = self.config.feature.date_column
        date_stats = df.agg(
            {date_col: "min", date_col: "max"}
        ).collect()[0]
        
        results = {
            "negative_costs": negative_costs,
            "zero_costs": zero_costs,
            "non_usd_currency": currency_check,
            "date_range": {
                "min": date_stats.get(f"min({date_col})"),
                "max": date_stats.get(f"max({date_col})")
            },
            "data_quality_issues": negative_costs > 0 or currency_check > 0
        }
        
        return results
    
    def validate_consistency(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data consistency (Section 3.1.4)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Consistency validation results
        """
        # Check for duplicate records
        total_records = df.count()
        distinct_records = df.distinct().count()
        duplicates = total_records - distinct_records
        
        # Check for data type consistency
        # (This would require schema validation)
        
        results = {
            "total_records": total_records,
            "distinct_records": distinct_records,
            "duplicate_records": duplicates,
            "duplicate_percentage": (duplicates / total_records * 100) if total_records > 0 else 0
        }
        
        return results
    
    def validate_timeliness(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data timeliness (Section 3.1.4)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Timeliness validation results
        """
        from datetime import datetime
        
        # Get latest date in data
        date_col = self.config.feature.date_column
        latest_date = df.agg({date_col: "max"}).collect()[0][f"max({date_col})"]
        
        if latest_date:
            # Calculate hours since last update
            hours_since_update = (datetime.now() - latest_date).total_seconds() / 3600
            
            # Check if data is fresh (within SLA)
            max_delay = self.config.data.max_data_delay_hours or 168
            is_fresh = hours_since_update <= max_delay
            
            results = {
                "latest_date": latest_date,
                "hours_since_update": hours_since_update,
                "is_fresh": is_fresh,
                "meets_sla": is_fresh
            }
        else:
            results = {
                "latest_date": None,
                "hours_since_update": None,
                "is_fresh": False,
                "meets_sla": False
            }
        
        return results
    
    def comprehensive_validation(self, df: DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Complete validation results
        """
        logger.info("Performing comprehensive data quality validation")
        
        validation_results = {
            "completeness": self.validate_completeness(df),
            "accuracy": self.validate_accuracy(df),
            "consistency": self.validate_consistency(df),
            "timeliness": self.validate_timeliness(df)
        }
        
        # Overall quality score
        quality_score = self._calculate_quality_score(validation_results)
        validation_results["quality_score"] = quality_score
        
        logger.info(f"Data quality score: {quality_score:.2f}%")
        return validation_results
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Quality score (0-100)
        """
        # Weighted average of different quality dimensions
        completeness = validation_results["completeness"]["completeness_rate"]
        accuracy = 100.0 if not validation_results["accuracy"]["data_quality_issues"] else 80.0
        consistency = 100.0 - min(validation_results["consistency"]["duplicate_percentage"], 10.0)
        timeliness = 100.0 if validation_results["timeliness"]["meets_sla"] else 70.0
        
        # Weighted average
        quality_score = (
            completeness * 0.3 +
            accuracy * 0.3 +
            consistency * 0.2 +
            timeliness * 0.2
        )
        
        return quality_score


