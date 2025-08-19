"""
Evaluation Metrics Calculation Module
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)
from typing import Dict, Union, Optional
from statsmodels.stats.stattools import durbin_watson
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Evaluation Metrics Calculator"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate various evaluation metrics for regression models
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights
            
        Returns:
            Dict: Dictionary containing various metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        
        # MAPE (handle zero values)
        mask = y_true != 0
        if mask.any():
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf
            
        # Use sklearn's MAPE (if available)
        try:
            metrics['sklearn_mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['sklearn_mape'] = metrics['mape']
        
        # Additional metrics
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # Error statistics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['median_error'] = np.median(errors)
        metrics['median_absolute_error'] = np.median(np.abs(errors))
        
        # Percentiles
        metrics['q25_error'] = np.percentile(np.abs(errors), 25)
        metrics['q75_error'] = np.percentile(np.abs(errors), 75)
        metrics['q95_error'] = np.percentile(np.abs(errors), 95)
        
        return metrics
    
    @staticmethod
    def calculate_metrics_by_segment(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    n_segments: int = 4) -> pd.DataFrame:
        """
        Calculate evaluation metrics by value segments
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_segments: Number of segments
            
        Returns:
            pd.DataFrame: Segmented evaluation results
        """
        # Create segments
        percentiles = np.linspace(0, 100, n_segments + 1)
        bins = np.percentile(y_true, percentiles)
        
        results = []
        for i in range(n_segments):
            if i < n_segments - 1:
                mask = (y_true >= bins[i]) & (y_true < bins[i+1])
                segment_name = f"[{bins[i]:.0f}, {bins[i+1]:.0f})"
            else:
                mask = y_true >= bins[i]
                segment_name = f"[{bins[i]:.0f}, ∞)"
            
            if mask.sum() > 0:
                segment_metrics = MetricsCalculator.calculate_regression_metrics(
                    y_true[mask], y_pred[mask]
                )
                segment_metrics['segment'] = f"Q{i+1}"
                segment_metrics['range'] = segment_name
                segment_metrics['count'] = mask.sum()
                results.append(segment_metrics)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_residual_diagnostics(y_true: np.ndarray,
                                      y_pred: np.ndarray) -> Dict:
        """
        Calculate residual diagnostic metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dict: Residual diagnostic results
        """
        residuals = y_true - y_pred
        
        diagnostics = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': pd.Series(residuals).skew(),
            'residual_kurtosis': pd.Series(residuals).kurtosis(),
            'residual_autocorrelation': pd.Series(residuals).autocorr() if len(residuals) > 1 else 0,
            'standardized_residuals': residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        }
        

        try:
            diagnostics['durbin_watson'] = durbin_watson(residuals)
        except:
            diagnostics['durbin_watson'] = None
        
        return diagnostics
    
    @staticmethod
    def format_metrics_report(metrics: Dict, 
                             model_name: str = "Model") -> str:
        """
        Format evaluation metrics report
        
        Args:
            metrics: Metrics dictionary
            model_name: Model name
            
        Returns:
            str: Formatted report
        """
        report = f"\n{'='*50}\n"
        report += f"{model_name} Evaluation Report\n"
        report += f"{'='*50}\n\n"
        
        report += "Main Metrics:\n"
        report += f"  MAE:  {metrics.get('mae', 0):.4f}\n"
        report += f"  RMSE: {metrics.get('rmse', 0):.4f}\n"
        report += f"  R²:   {metrics.get('r2', 0):.4f}\n"
        report += f"  MAPE: {metrics.get('mape', 0):.2f}%\n"
        
        report += "\nError Distribution:\n"
        report += f"  Mean Error:     {metrics.get('mean_error', 0):.4f}\n"
        report += f"  Error Std Dev:  {metrics.get('std_error', 0):.4f}\n"
        report += f"  Median Error:   {metrics.get('median_error', 0):.4f}\n"
        
        report += "\nError Percentiles:\n"
        report += f"  25%: {metrics.get('q25_error', 0):.4f}\n"
        report += f"  75%: {metrics.get('q75_error', 0):.4f}\n"
        report += f"  95%: {metrics.get('q95_error', 0):.4f}\n"
        
        report += f"\n{'='*50}\n"
        
        return report