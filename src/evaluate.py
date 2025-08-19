#!/usr/bin/env python
"""
模型评估脚本
使用命令: uv run evaluate
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import *
from visualization.visualizer import Visualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.inf
    
    # 额外指标
    max_error = np.max(np.abs(y_true - y_pred))
    median_error = np.median(np.abs(y_true - y_pred))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'max_error': max_error,
        'median_error': median_error
    }

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("开始模型评估...")
    logger.info("=" * 50)
    
    # 1. 检查必要文件
    logger.info("\n1. 检查必要文件...")
    
    required_files = {
        'model': MODEL_OUTPUT_DIR / 'best_model.pkl',
        'scaler': MODEL_OUTPUT_DIR / 'scaler.pkl',
        'test_predictions': MODEL_OUTPUT_DIR / 'test_predictions.csv',
        'model_config': MODEL_OUTPUT_DIR / 'model_config.json',
        'training_results': MODEL_OUTPUT_DIR / 'training_results.json'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not path.exists():
            missing_files.append(name)
            logger.warning(f"缺少文件: {path}")
    
    if missing_files:
        logger.error(f"缺少必要文件，请先运行 'uv run model-train'")
        return
    
    # 2. 加载模型和配置
    logger.info("\n2. 加载模型和配置...")
    
    # 加载最佳模型
    best_model = joblib.load(required_files['model'])
    logger.info("最佳模型已加载")
    
    # 加载模型配置
    with open(required_files['model_config'], 'r') as f:
        model_config = json.load(f)
    logger.info(f"模型类型: {model_config.get('best_model', 'Unknown')}")
    
    # 加载训练结果
    with open(required_files['training_results'], 'r') as f:
        training_results = json.load(f)
    
    # 3. 加载测试集预测结果
    logger.info("\n3. 加载测试集预测结果...")
    test_predictions = pd.read_csv(required_files['test_predictions'])
    
    y_true = test_predictions['actual'].values
    y_pred = test_predictions['predicted'].values
    
    # 4. 详细评估
    logger.info("\n4. 详细评估...")
    
    # 计算所有指标
    metrics = calculate_metrics(y_true, y_pred)
    
    logger.info("测试集评估指标:")
    logger.info(f"  MAE: {metrics['mae']:.2f}")
    logger.info(f"  RMSE: {metrics['rmse']:.2f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  最大误差: {metrics['max_error']:.2f}")
    logger.info(f"  中位数误差: {metrics['median_error']:.2f}")
    
    # 5. 误差分析
    logger.info("\n5. 误差分析...")
    
    errors = np.array(y_true) - np.array(y_pred)
    error_stats = {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'q25_error': np.percentile(np.abs(errors), 25),
        'q50_error': np.percentile(np.abs(errors), 50),
        'q75_error': np.percentile(np.abs(errors), 75),
        'q95_error': np.percentile(np.abs(errors), 95)
    }
    
    logger.info("误差分布:")
    logger.info(f"  平均误差: {error_stats['mean_error']:.2f}")
    logger.info(f"  误差标准差: {error_stats['std_error']:.2f}")
    logger.info(f"  25%分位数: {error_stats['q25_error']:.2f}")
    logger.info(f"  50%分位数: {error_stats['q50_error']:.2f}")
    logger.info(f"  75%分位数: {error_stats['q75_error']:.2f}")
    logger.info(f"  95%分位数: {error_stats['q95_error']:.2f}")
    
    # 保存误差分析
    error_analysis_path = EVALUATION_OUTPUT_DIR / 'error_analysis.json'
    with open(error_analysis_path, 'w') as f:
        json.dump(error_stats, f, indent=2)
    logger.info(f"误差分析已保存至: {error_analysis_path}")
    
    # 6. 分段评估
    logger.info("\n6. 分段评估...")
    
    # 按预测值大小分段
    n_bins = 4
    bins = np.percentile(np.array(y_true), np.linspace(0, 100, n_bins + 1))
    bin_labels = [f'Q{i+1}' for i in range(n_bins)]
    
    segment_results = []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        else:
            mask = y_true >= bins[i]
        
        if mask.sum() > 0:
            segment_metrics = calculate_metrics(y_true[mask], y_pred[mask])
            segment_metrics['segment'] = bin_labels[i]
            segment_metrics['count'] = mask.sum()
            segment_metrics['range'] = f"[{bins[i]:.0f}, {bins[i+1]:.0f})"
            segment_results.append(segment_metrics)
            
            logger.info(f"  {bin_labels[i]} {segment_metrics['range']}: "
                       f"n={segment_metrics['count']}, "
                       f"R²={segment_metrics['r2']:.3f}, "
                       f"RMSE={segment_metrics['rmse']:.0f}")
    
    # 保存分段评估结果
    segment_df = pd.DataFrame(segment_results)
    segment_path = EVALUATION_OUTPUT_DIR / 'segment_evaluation.csv'
    segment_df.to_csv(segment_path, index=False)
    logger.info(f"分段评估结果已保存至: {segment_path}")
    
    # 7. 模型比较分析
    logger.info("\n7. 模型比较分析...")
    
    model_comparison = []
    for model_name, results in training_results.items():
        comparison_entry = {
            'model': model_name,
            'train_r2': results['train_metrics']['r2'],
            'val_r2': results['val_metrics']['r2'],
            'train_rmse': results['train_metrics']['rmse'],
            'val_rmse': results['val_metrics']['rmse'],
            'overfit_ratio': results['train_metrics']['r2'] / results['val_metrics']['r2'] if results['val_metrics']['r2'] > 0 else np.inf
        }
        model_comparison.append(comparison_entry)
    
    comparison_df = pd.DataFrame(model_comparison)
    comparison_df = comparison_df.sort_values('val_r2', ascending=False)
    
    comparison_path = EVALUATION_OUTPUT_DIR / 'model_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"模型比较结果已保存至: {comparison_path}")
    
    # 打印前3个模型
    logger.info("Top 3 模型:")
    for idx, row in comparison_df.head(3).iterrows():
        logger.info(f"  {row['model']}: Val R²={row['val_r2']:.4f}, "
                   f"过拟合率={row['overfit_ratio']:.2f}")
    
    # 8. 生成评估可视化
    logger.info("\n8. 生成评估可视化...")
    viz = Visualizer(EVALUATION_OUTPUT_DIR)
    
    # 预测vs实际散点图（带误差带）
    viz.plot_predictions(y_true, y_pred, model_name='Final_Evaluation')
    
    # 残差分析图
    viz.plot_residuals(y_true, y_pred, model_name='Final_Evaluation')
    
    # 9. 生成最终评估报告
    logger.info("\n9. 生成最终评估报告...")
    
    final_report = {
        'model_info': {
            'type': model_config.get('best_model', 'Unknown'),
            'n_features': model_config.get('n_features', 0),
            'n_samples': model_config.get('n_samples', 0),
            'train_size': model_config.get('train_size', 0),
            'val_size': model_config.get('val_size', 0),
            'test_size': model_config.get('test_size', 0)
        },
        'test_metrics': metrics,
        'error_statistics': error_stats,
        'performance_summary': {
            'excellent': metrics['r2'] > 0.9,
            'good': 0.8 <= metrics['r2'] <= 0.9,
            'acceptable': 0.7 <= metrics['r2'] < 0.8,
            'poor': metrics['r2'] < 0.7,
            'mape_acceptable': metrics['mape'] < 20
        }
    }
    
    report_path = EVALUATION_OUTPUT_DIR / 'final_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    logger.info(f"最终评估报告已保存至: {report_path}")
    
    # 10. 生成Markdown报告
    logger.info("\n10. 生成Markdown报告...")
    
    markdown_report = f"""# 污水处理厂能耗预测模型评估报告

## 1. 模型信息
- **模型类型**: {model_config.get('best_model', 'Unknown')}
- **特征数量**: {model_config.get('n_features', 0)}
- **总样本数**: {model_config.get('n_samples', 0)}
- **训练集大小**: {model_config.get('train_size', 0)}
- **验证集大小**: {model_config.get('val_size', 0)}
- **测试集大小**: {model_config.get('test_size', 0)}

## 2. 测试集性能指标
| 指标 | 值 |
|------|-----|
| MAE | {metrics['mae']:.2f} |
| RMSE | {metrics['rmse']:.2f} |
| R² | {metrics['r2']:.4f} |
| MAPE | {metrics['mape']:.2f}% |
| 最大误差 | {metrics['max_error']:.2f} |
| 中位数误差 | {metrics['median_error']:.2f} |

## 3. 误差分布
| 分位数 | 绝对误差 |
|---------|----------|
| 25% | {error_stats['q25_error']:.2f} |
| 50% | {error_stats['q50_error']:.2f} |
| 75% | {error_stats['q75_error']:.2f} |
| 95% | {error_stats['q95_error']:.2f} |

## 4. 性能评价
- **R² 得分**: {'优秀 (>0.9)' if metrics['r2'] > 0.9 else '良好 (0.8-0.9)' if metrics['r2'] >= 0.8 else '可接受 (0.7-0.8)' if metrics['r2'] >= 0.7 else '需要改进 (<0.7)'}
- **MAPE**: {'可接受 (<20%)' if metrics['mape'] < 20 else '偏高 (≥20%)'}

## 5. 结论
模型在测试集上的R²为{metrics['r2']:.4f}，RMSE为{metrics['rmse']:.2f}，表明模型具有{'良好的' if metrics['r2'] > 0.8 else '一定的'}预测能力。
平均绝对百分比误差(MAPE)为{metrics['mape']:.2f}%，{'满足' if metrics['mape'] < 20 else '略高于'}工程应用要求。

## 6. 改进建议
1. {'模型性能良好，可以考虑部署使用' if metrics['r2'] > 0.85 else '考虑收集更多数据或尝试其他特征工程方法'}
2. {'关注高能耗样本的预测精度' if metrics['mape'] > 15 else '模型预测稳定'}
3. 定期更新模型以适应新的运行条件
"""
    
    markdown_path = EVALUATION_OUTPUT_DIR / 'evaluation_report.md'
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    logger.info(f"Markdown报告已保存至: {markdown_path}")
    
    logger.info("\n" + "=" * 50)
    logger.info("模型评估完成！")
    logger.info(f"最终R²: {metrics['r2']:.4f}")
    logger.info(f"最终RMSE: {metrics['rmse']:.2f}")
    logger.info(f"所有结果已保存至: {EVALUATION_OUTPUT_DIR}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()