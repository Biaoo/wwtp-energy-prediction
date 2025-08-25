#!/usr/bin/env python
"""
模型训练脚本
使用命令: uv run model-train
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import *
from data.data_loader import DataLoader
from data.data_cleaner import DataCleaner
from features.feature_engineering import FeatureEngineer
from models.model_trainer import ModelTrainer
from visualization.visualizer import Visualizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("开始模型训练...")
    logger.info("=" * 50)
    
    # 1. 加载和准备数据
    logger.info("\n1. 加载和准备数据...")
    
    # 检查是否存在处理后的数据
    processed_data_path = ANALYSIS_OUTPUT_DIR / 'processed_data.csv'
    if processed_data_path.exists():
        logger.info(f"加载已处理的数据: {processed_data_path}")
        data = pd.read_csv(processed_data_path)
    else:
        logger.info("未找到处理后的数据，重新处理...")
        # 加载原始数据
        loader = DataLoader(RAW_DATA_PATH)
        data = loader.load_data()
        
        # 数据清洗
        cleaner = DataCleaner(data)
        data = cleaner.clean_data(
            drop_geographic=True,
            geographic_cols=GEOGRAPHIC_FEATURES,
            handle_missing=True,
            missing_strategy='median',
            handle_special=True,
            remove_outliers=False
        )
        
        # 特征工程
        engineer = FeatureEngineer(data)
        data = engineer.create_derived_features()
    
    # 2. 准备特征和目标变量
    logger.info("\n2. 准备特征和目标变量...")
    
    # 分离特征和目标
    if TARGET_COLUMN not in data.columns:
        logger.error(f"目标变量 {TARGET_COLUMN} 不存在!")
        return
    
    # 移除ID列和目标列
    feature_cols = [col for col in data.columns 
                   if col not in [ID_COLUMN, TARGET_COLUMN]]
    
    # 对类别特征进行编码
    categorical_features = []
    for col in PROCESS_FEATURES:
        if col in feature_cols:
            categorical_features.append(col)
    
    if categorical_features:
        logger.info(f"编码类别特征: {categorical_features}")
        
        # 替换NaN值为'Unknown'
        for col in categorical_features:
            data[col] = data[col].fillna('Unknown')
        
        engineer = FeatureEngineer(data[feature_cols + [TARGET_COLUMN]])
        data_encoded = engineer.encode_categorical_features(
            categorical_features, 
            encoding_type='onehot'
        )
        
        # 更新特征列 - 包含所有特征（数值和编码后的类别）
        feature_cols = [col for col in data_encoded.columns if col != TARGET_COLUMN]
        X = data_encoded[feature_cols]
        y = data_encoded[TARGET_COLUMN]
        
        # 保存编码器（如果有的话）
        if hasattr(engineer, 'encoders'):
            encoder_path = MODEL_OUTPUT_DIR / 'encoders.pkl'
            joblib.dump(engineer.encoders, encoder_path)
            logger.info(f"类别编码器已保存至: {encoder_path}")
    else:
        X = data[feature_cols]
        y = data[TARGET_COLUMN]
    
    # 重要改动：不再过滤只保留数值型特征，而是保留所有特征
    # 确保所有特征都是数值型（one-hot编码后的特征已经是数值型）
    
    logger.info(f"特征数量: {X.shape[1]}")
    logger.info(f"样本数量: {X.shape[0]}")
    
    # 检查是否包含工艺特征
    process_related_cols = [col for col in X.columns if any(
        proc in col.lower() for proc in ['process', 'a2o', 'ao_', 'sbr', 'mbr', 'oxidation', 
                                         'biofilm', 'membrane', 'filtration', 'chlorine', 'uv']
    )]
    if process_related_cols:
        logger.info(f"包含 {len(process_related_cols)} 个工艺相关特征")
        logger.info(f"工艺特征示例: {process_related_cols[:5]}")
    else:
        logger.warning("警告：没有找到工艺相关特征！")
    
    # 保存特征名称
    feature_names_path = MODEL_OUTPUT_DIR / 'feature_names.json'
    with open(feature_names_path, 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    logger.info(f"特征名称已保存至: {feature_names_path}")
    
    # 3. 数据集划分
    logger.info("\n3. 数据集划分...")
    trainer = ModelTrainer(random_state=RANDOM_SEED)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
        X, y, 
        test_size=TEST_SIZE,
        val_size=VALIDATION_SIZE
    )
    
    # 特征缩放
    logger.info("进行特征缩放...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        np.array(scaler.transform(X_val)),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        np.array(scaler.transform(X_test)),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 保存缩放器
    scaler_path = MODEL_OUTPUT_DIR / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"特征缩放器已保存至: {scaler_path}")
    
    # 4. 训练基准模型
    logger.info("\n4. 训练基准模型...")
    baseline_results = trainer.train_baseline_models(
        X_train_scaled, y_train, 
        X_val_scaled, y_val
    )
    
    # 5. 训练树模型
    logger.info("\n5. 训练树模型...")
    tree_results = trainer.train_tree_models(
        X_train, y_train,  # 树模型不需要缩放
        X_val, y_val
    )
    
    # 6. 选择最佳模型
    logger.info("\n6. 选择最佳模型...")
    best_model = trainer.select_best_model(metric='r2')
    
    # 7. 在测试集上评估最佳模型
    logger.info("\n7. 在测试集上评估最佳模型...")
    
    # 根据模型类型选择数据
    best_model_name = None
    for name, model in trainer.models.items():
        if model == best_model:
            best_model_name = name
            break
    
    if best_model_name and best_model_name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
        # 线性模型使用缩放后的数据
        y_test_pred = best_model.predict(X_test_scaled)
    else:
        # 树模型使用原始数据
        y_test_pred = best_model.predict(X_test)
    
    test_metrics = trainer.evaluate_model(y_test, y_test_pred)
    logger.info(f"测试集性能:")
    logger.info(f"  MAE: {test_metrics['mae']:.2f}")
    logger.info(f"  RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
    
    # 8. 保存模型和结果
    logger.info("\n8. 保存模型和结果...")
    
    # 保存最佳模型
    best_model_path = MODEL_OUTPUT_DIR / 'best_model.pkl'
    trainer.save_model(best_model, best_model_path)
    
    # 保存所有模型（可选）
    for name, model in trainer.models.items():
        model_path = MODEL_OUTPUT_DIR / f'{name}_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"模型 {name} 已保存至: {model_path}")
    
    # 保存训练结果
    results_summary = {}
    for name, result in trainer.results.items():
        results_summary[name] = {
            'train_metrics': result['train_metrics'],
            'val_metrics': result['val_metrics']
        }
    
    results_path = MODEL_OUTPUT_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    logger.info(f"训练结果已保存至: {results_path}")
    
    # 保存测试集预测结果
    test_predictions = pd.DataFrame({
        'actual': y_test,
        'predicted': y_test_pred,
        'error': y_test - y_test_pred,
        'error_percentage': ((y_test - y_test_pred) / y_test * 100)
    })
    test_pred_path = MODEL_OUTPUT_DIR / 'test_predictions.csv'
    test_predictions.to_csv(test_pred_path, index=False)
    logger.info(f"测试集预测结果已保存至: {test_pred_path}")
    
    # 9. 生成可视化
    logger.info("\n9. 生成模型比较可视化...")
    viz = Visualizer(MODEL_OUTPUT_DIR)
    
    # 模型比较图
    viz.plot_model_comparison(trainer.results)
    
    # 最佳模型的预测图
    viz.plot_predictions(y_test, y_test_pred, model_name=best_model_name or 'Best')
    
    # 残差图
    viz.plot_residuals(y_test, y_test_pred, model_name=best_model_name or 'Best')
    
    # 特征重要性（如果是树模型）
    if best_model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if 'feature_importance' in trainer.results[best_model_name]:
            viz.plot_feature_importance(
                trainer.results[best_model_name]['feature_importance'],
                model_name=best_model_name or 'Best',
                top_n=20
            )
            
            # 分析工艺特征的重要性
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # 显示工艺特征的重要性
                if process_related_cols:
                    process_importance = feature_importance[
                        feature_importance['feature'].isin(process_related_cols)
                    ]
                    if not process_importance.empty:
                        logger.info("\n工艺特征重要性:")
                        for _, row in process_importance.head(10).iterrows():
                            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 10. 保存模型配置
    model_config = {
        'best_model': best_model_name,
        'test_metrics': test_metrics,
        'feature_columns': list(X.columns),
        'target_column': TARGET_COLUMN,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'includes_process_features': len(process_related_cols) > 0 if 'process_related_cols' in locals() else False,
        'n_process_features': len(process_related_cols) if 'process_related_cols' in locals() else 0
    }
    
    config_path = MODEL_OUTPUT_DIR / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    logger.info(f"模型配置已保存至: {config_path}")
    
    logger.info("\n" + "=" * 50)
    logger.info("模型训练完成！")
    logger.info(f"最佳模型: {best_model_name}")
    logger.info(f"测试集 R²: {test_metrics['r2']:.4f}")
    logger.info(f"所有结果已保存至: {MODEL_OUTPUT_DIR}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()