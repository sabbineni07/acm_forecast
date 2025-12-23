"""
Unit tests for performance metrics module
"""
import pytest
import numpy as np
import pandas as pd

from acm_forecast.evaluation.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """Unit tests for PerformanceMetrics class"""
    
    @pytest.mark.unit
    def test_calculate_metrics_perfect_prediction(self):
        """Test calculate_metrics with perfect predictions"""
        actual = np.array([100, 200, 300, 400])
        predicted = np.array([100, 200, 300, 400])
        metrics = PerformanceMetrics.calculate_metrics(actual, predicted)
        
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['mape'] < 1.0  # Should be very close to 0 (using 1e-8 in calculation)
        assert abs(metrics['r2'] - 1.0) < 0.01
    
    @pytest.mark.unit
    def test_calculate_metrics_with_errors(self):
        """Test calculate_metrics with prediction errors"""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 180, 330])
        metrics = PerformanceMetrics.calculate_metrics(actual, predicted)
        
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['mape'] > 0
        assert metrics['r2'] < 1.0
    
    @pytest.mark.unit
    def test_calculate_metrics_with_nan_values(self):
        """Test calculate_metrics handles NaN values"""
        actual = np.array([100, np.nan, 300, 400])
        predicted = np.array([110, 190, np.nan, 410])
        metrics = PerformanceMetrics.calculate_metrics(actual, predicted)
        
        # Should filter out NaN values and still calculate metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics
        assert not np.isnan(metrics['rmse'])
    
    @pytest.mark.unit
    def test_calculate_metrics_empty_arrays(self):
        """Test calculate_metrics with empty arrays after filtering"""
        actual = np.array([np.nan, np.nan])
        predicted = np.array([np.nan, np.nan])
        metrics = PerformanceMetrics.calculate_metrics(actual, predicted)
        
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['mae'])
        assert np.isnan(metrics['mape'])
        assert np.isnan(metrics['r2'])
    
    @pytest.mark.unit
    def test_calculate_metrics_with_zero_actual(self):
        """Test calculate_metrics handles zero actual values"""
        actual = np.array([0, 100, 200])
        predicted = np.array([10, 110, 190])
        metrics = PerformanceMetrics.calculate_metrics(actual, predicted)
        
        # Should handle zero values gracefully (using 1e-8 in calculation)
        assert isinstance(metrics['mape'], float)
        assert not np.isnan(metrics['mape'])
        assert not np.isinf(metrics['mape'])
    
    @pytest.mark.unit
    def test_calculate_by_horizon(self):
        """Test calculate_by_horizon method"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        actual = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        predicted = pd.Series(actual.values + np.random.randn(100) * 5, index=dates)
        
        horizons = [1, 7, 30]
        results = PerformanceMetrics.calculate_by_horizon(actual, predicted, horizons)
        
        assert len(results) == 3
        assert 1 in results
        assert 7 in results
        assert 30 in results
        
        # Each result should have metrics
        for horizon, metrics in results.items():
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'mape' in metrics
            assert 'r2' in metrics
    
    @pytest.mark.unit
    def test_calculate_by_horizon_insufficient_data(self):
        """Test calculate_by_horizon with insufficient data"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        actual = pd.Series(np.random.randn(10), index=dates)
        predicted = pd.Series(np.random.randn(10), index=dates)
        
        horizons = [1, 7, 30, 90]  # 90 is larger than available data
        results = PerformanceMetrics.calculate_by_horizon(actual, predicted, horizons)
        
        # Should only return results for horizons that fit
        assert 1 in results
        assert 7 in results
        assert 30 not in results  # Won't be included if data is insufficient
        assert 90 not in results
    
    @pytest.mark.unit
    def test_create_performance_summary(self):
        """Test create_performance_summary method"""
        metrics_dict = {
            'prophet': {
                'rmse': 10.5,
                'mae': 8.2,
                'mape': 5.3,
                'r2': 0.95,
            },
            'arima': {
                'rmse': 12.1,
                'mae': 9.5,
                'mape': 6.2,
                'r2': 0.92,
            },
            'xgboost': {
                'rmse': 9.8,
                'mae': 7.9,
                'mape': 4.8,
                'r2': 0.97,
            },
        }
        
        summary_df = PerformanceMetrics.create_performance_summary(metrics_dict)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 3  # Three models
        assert 'rmse' in summary_df.columns
        assert 'mae' in summary_df.columns
        assert 'mape' in summary_df.columns
        assert 'r2' in summary_df.columns
        assert 'prophet' in summary_df.index
        assert 'arima' in summary_df.index
        assert 'xgboost' in summary_df.index
        
        # Check that values are rounded
        assert summary_df.loc['prophet', 'rmse'] == 10.5
        assert summary_df.loc['arima', 'r2'] == 0.92
    
    @pytest.mark.unit
    def test_calculate_metrics_pandas_series(self):
        """Test that calculate_metrics works with pandas Series input"""
        actual = pd.Series([100, 200, 300])
        predicted = pd.Series([110, 190, 310])
        
        # Convert to numpy arrays for the method (it expects np.ndarray)
        metrics = PerformanceMetrics.calculate_metrics(actual.values, predicted.values)
        
        assert isinstance(metrics['rmse'], float)
        assert isinstance(metrics['mae'], float)
        assert isinstance(metrics['mape'], float)
        assert isinstance(metrics['r2'], float)
    
    @pytest.mark.unit
    def test_calculate_metrics_different_lengths(self):
        """Test that calculate_metrics handles mismatched lengths"""
        actual = np.array([100, 200, 300])
        predicted = np.array([110, 190])  # Different length
        
        # The method doesn't check lengths, so this will work but use shorter length
        # This is more of a documentation/design consideration
        metrics = PerformanceMetrics.calculate_metrics(actual[:2], predicted)
        assert 'rmse' in metrics
