"""
Visualization Module
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualizer"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_data_distribution(self, data: pd.DataFrame, numeric_cols: List[str]):
        """
        Plot data distribution
        
        Args:
            data: Data
            numeric_cols: Numeric columns
        """
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                axes[idx].hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                
        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / 'data_distributions.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存对应的数据
        stats_df = data[numeric_cols].describe()
        csv_path = self.output_dir / 'data_distributions_stats.csv'
        stats_df.to_csv(csv_path)
        
        logger.info(f"Data distribution plot saved to: {fig_path}")
        logger.info(f"Statistical data saved to: {csv_path}")
        
    def plot_correlation_matrix(self, data: pd.DataFrame, figsize: Tuple = (12, 10)):
        """
        Plot correlation matrix
        
        Args:
            data: Data
            figsize: Figure size
        """
        # 计算相关性矩阵
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        # 绘制热图
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / 'correlation_matrix.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存相关性数据
        csv_path = self.output_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(csv_path)
        
        logger.info(f"相关性矩阵图已保存至: {fig_path}")
        logger.info(f"相关性数据已保存至: {csv_path}")
        
    def plot_target_correlations(self, data: pd.DataFrame, target_col: str, top_n: int = 20):
        """
        绘制与目标变量的相关性
        
        Args:
            data: 数据
            target_col: 目标变量列名
            top_n: 显示前N个特征
        """
        # 计算与目标变量的相关性
        numeric_data = data.select_dtypes(include=[np.number])
        correlations = numeric_data.corrwith(data[target_col]).abs().sort_values(ascending=False)
        
        # 移除目标变量自身
        correlations = correlations[correlations.index != target_col]
        
        # 选择前N个
        top_correlations = correlations.head(top_n)
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        top_correlations.plot(kind='barh')
        plt.xlabel('Absolute Correlation')
        plt.title(f'Top {top_n} Features Correlated with {target_col}')
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / 'target_correlations.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        csv_path = self.output_dir / 'target_correlations.csv'
        correlations.to_csv(csv_path, header=['correlation'])
        
        logger.info(f"目标相关性图已保存至: {fig_path}")
        logger.info(f"相关性数据已保存至: {csv_path}")
        
    def plot_model_comparison(self, results: Dict):
        """
        绘制模型比较图
        
        Args:
            results: 模型结果字典
        """
        # 准备数据
        model_names = []
        train_r2 = []
        val_r2 = []
        train_rmse = []
        val_rmse = []
        
        for name, result in results.items():
            model_names.append(name)
            train_r2.append(result['train_metrics']['r2'])
            val_r2.append(result['val_metrics']['r2'])
            train_rmse.append(result['train_metrics']['rmse'])
            val_rmse.append(result['val_metrics']['rmse'])
            
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² 比较
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0].bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
        axes[0].bar(x + width/2, val_r2, width, label='Validation', alpha=0.8)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Model R² Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMSE 比较
        axes[1].bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
        axes[1].bar(x + width/2, val_rmse, width, label='Validation', alpha=0.8)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Model RMSE Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / 'model_comparison.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        comparison_df = pd.DataFrame({
            'model': model_names,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        })
        csv_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        
        logger.info(f"模型比较图已保存至: {fig_path}")
        logger.info(f"模型比较数据已保存至: {csv_path}")
        
    def plot_predictions(self, y_true, y_pred, model_name: str = 'Model'):
        """
        绘制预测vs实际值图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
        """
        plt.figure(figsize=(8, 6))
        
        # 散点图
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # 添加理想预测线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # 计算R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} - Predictions vs Actual (R² = {r2:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f'{model_name.lower()}_predictions.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存预测数据
        pred_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'error': y_true - y_pred,
            'error_percentage': ((y_true - y_pred) / y_true * 100) if (y_true != 0).all() else 0
        })
        csv_path = self.output_dir / f'{model_name.lower()}_predictions.csv'
        pred_df.to_csv(csv_path, index=False)
        
        logger.info(f"预测图已保存至: {fig_path}")
        logger.info(f"预测数据已保存至: {csv_path}")
        
    def plot_residuals(self, y_true, y_pred, model_name: str = 'Model'):
        """
        绘制残差图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 残差分布图
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residual Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f'{model_name.lower()}_residuals.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"残差图已保存至: {fig_path}")
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str = 'Model', top_n: int = 20):
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            model_name: 模型名称
            top_n: 显示前N个特征
        """
        # 选择前N个重要特征
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'].tolist())
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # 保存图片
        fig_path = self.output_dir / f'{model_name.lower()}_feature_importance.png'
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 保存数据
        csv_path = self.output_dir / f'{model_name.lower()}_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        
        logger.info(f"特征重要性图已保存至: {fig_path}")
        logger.info(f"特征重要性数据已保存至: {csv_path}")