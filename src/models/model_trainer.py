"""
模型训练模块
"""
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost未安装，将跳过XGBoost模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM未安装，将跳过LightGBM模型")

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化模型训练器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_params = {}
        self.results = {}
        
    def split_data(self, 
                   X: pd.DataFrame, 
                   y: pd.Series,
                   test_size: float = 0.2,
                   val_size: float = 0.2) -> Tuple:
        """
        划分数据集
        
        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            val_size: 验证集比例（从训练集中划分）
            
        Returns:
            Tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 首先划分训练集和测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # 从训练集中划分验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {X_train.shape}")
        logger.info(f"  验证集: {X_val.shape}")
        logger.info(f"  测试集: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_baseline_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """
        训练基准模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            Dict: 模型结果
        """
        logger.info("训练基准模型...")
        
        baseline_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'elastic_net': ElasticNet(alpha=0.1, random_state=self.random_state)
        }
        
        for name, model in baseline_models.items():
            logger.info(f"训练 {name}...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # 评估
            train_metrics = self.evaluate_model(y_train, y_train_pred)
            val_metrics = self.evaluate_model(y_val, y_val_pred)
            
            self.models[name] = model
            self.results[name] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model': model
            }
            
            logger.info(f"{name} - 验证集 R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.2f}")
            
        return self.results
        
    def train_tree_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """
        训练树模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            
        Returns:
            Dict: 模型结果
        """
        logger.info("训练树模型...")
        
        # 随机森林
        logger.info("训练随机森林...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        y_train_pred = rf_model.predict(X_train)
        y_val_pred = rf_model.predict(X_val)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'train_metrics': self.evaluate_model(y_train, y_train_pred),
            'val_metrics': self.evaluate_model(y_val, y_val_pred),
            'model': rf_model,
            'feature_importance': pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        logger.info(f"随机森林 - 验证集 R²: {self.results['random_forest']['val_metrics']['r2']:.4f}")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("训练XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=self.random_state
            )
            xgb_model.fit(X_train, y_train)
            
            y_train_pred = xgb_model.predict(X_train)
            y_val_pred = xgb_model.predict(X_val)
            
            self.models['xgboost'] = xgb_model
            self.results['xgboost'] = {
                'train_metrics': self.evaluate_model(y_train, y_train_pred),
                'val_metrics': self.evaluate_model(y_val, y_val_pred),
                'model': xgb_model,
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            
            logger.info(f"XGBoost - 验证集 R²: {self.results['xgboost']['val_metrics']['r2']:.4f}")
            
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("训练LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            
            y_train_pred = lgb_model.predict(X_train)
            y_val_pred = lgb_model.predict(X_val)
            
            self.models['lightgbm'] = lgb_model
            self.results['lightgbm'] = {
                'train_metrics': self.evaluate_model(y_train, y_train_pred),
                'val_metrics': self.evaluate_model(y_val, y_val_pred),
                'model': lgb_model,
                'feature_importance': pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': lgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            
            logger.info(f"LightGBM - 验证集 R²: {self.results['lightgbm']['val_metrics']['r2']:.4f}")
            
        return self.results
        
    def hyperparameter_tuning(self, 
                            X_train, y_train,
                            model_name: str = 'random_forest',
                            param_grid: Optional[Dict] = None,
                            cv: int = 5) -> Dict:
        """
        超参数调优
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            model_name: 模型名称
            param_grid: 参数网格
            cv: 交叉验证折数
            
        Returns:
            Dict: 最佳参数和模型
        """
        logger.info(f"对 {model_name} 进行超参数调优...")
        
        if model_name == 'random_forest':
            base_model = RandomForestRegressor(random_state=self.random_state)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            base_model = xgb.XGBRegressor(random_state=self.random_state)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8]
                }
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            base_model = lgb.LGBMRegressor(random_state=self.random_state, verbosity=-1)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50],
                    'feature_fraction': [0.7, 0.9]
                }
        else:
            logger.warning(f"模型 {model_name} 不支持超参数调优")
            return {}
            
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params[model_name] = grid_search.best_params_
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
    def evaluate_model(self, y_true, y_pred) -> Dict:
        """
        评估模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict: 评估指标
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else np.inf
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
    def select_best_model(self, metric: str = 'r2') -> Any:
        """
        选择最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型
        """
        best_score = -np.inf if metric in ['r2'] else np.inf
        best_model_name = None
        
        for name, result in self.results.items():
            score = result['val_metrics'][metric]
            
            if metric in ['r2']:
                if score > best_score:
                    best_score = score
                    best_model_name = name
            else:
                if score < best_score:
                    best_score = score
                    best_model_name = name
                    
        if best_model_name:
            self.best_model = self.models[best_model_name]
            logger.info(f"最佳模型: {best_model_name}, {metric}: {best_score:.4f}")
            
        return self.best_model
        
    def save_model(self, model, filepath: Path):
        """
        保存模型
        
        Args:
            model: 要保存的模型
            filepath: 保存路径
        """
        joblib.dump(model, filepath)
        logger.info(f"模型已保存至: {filepath}")
        
    def load_model(self, filepath: Path):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型
        """
        model = joblib.load(filepath)
        logger.info(f"模型已从 {filepath} 加载")
        return model