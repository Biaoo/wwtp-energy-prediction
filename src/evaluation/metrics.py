"""
评估指标计算模块
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
    """评估指标计算器"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        计算回归模型的各种评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            sample_weight: 样本权重
            
        Returns:
            Dict: 包含各种指标的字典
        """
        metrics = {}
        
        # 基础指标
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        
        # MAPE (处理零值)
        mask = y_true != 0
        if mask.any():
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf
            
        # 使用sklearn的MAPE（如果可用）
        try:
            metrics['sklearn_mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['sklearn_mape'] = metrics['mape']
        
        # 额外指标
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
        metrics['max_error'] = max_error(y_true, y_pred)
        
        # 误差统计
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['median_error'] = np.median(errors)
        metrics['median_absolute_error'] = np.median(np.abs(errors))
        
        # 分位数
        metrics['q25_error'] = np.percentile(np.abs(errors), 25)
        metrics['q75_error'] = np.percentile(np.abs(errors), 75)
        metrics['q95_error'] = np.percentile(np.abs(errors), 95)
        
        return metrics
    
    @staticmethod
    def calculate_metrics_by_segment(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    n_segments: int = 4) -> pd.DataFrame:
        """
        按数值段计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            n_segments: 分段数量
            
        Returns:
            pd.DataFrame: 分段评估结果
        """
        # 创建分段
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
        计算残差诊断指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict: 残差诊断结果
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
        格式化评估指标报告
        
        Args:
            metrics: 指标字典
            model_name: 模型名称
            
        Returns:
            str: 格式化的报告
        """
        report = f"\n{'='*50}\n"
        report += f"{model_name} 评估报告\n"
        report += f"{'='*50}\n\n"
        
        report += "主要指标:\n"
        report += f"  MAE:  {metrics.get('mae', 0):.4f}\n"
        report += f"  RMSE: {metrics.get('rmse', 0):.4f}\n"
        report += f"  R²:   {metrics.get('r2', 0):.4f}\n"
        report += f"  MAPE: {metrics.get('mape', 0):.2f}%\n"
        
        report += "\n误差分布:\n"
        report += f"  平均误差:   {metrics.get('mean_error', 0):.4f}\n"
        report += f"  误差标准差: {metrics.get('std_error', 0):.4f}\n"
        report += f"  中位数误差: {metrics.get('median_error', 0):.4f}\n"
        
        report += "\n误差分位数:\n"
        report += f"  25%: {metrics.get('q25_error', 0):.4f}\n"
        report += f"  75%: {metrics.get('q75_error', 0):.4f}\n"
        report += f"  95%: {metrics.get('q95_error', 0):.4f}\n"
        
        report += f"\n{'='*50}\n"
        
        return report