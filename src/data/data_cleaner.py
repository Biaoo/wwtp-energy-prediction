"""
数据清洗模块
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化数据清洗器
        
        Args:
            data: 原始数据
        """
        self.data = data.copy()
        
    def drop_geographic_features(self, geographic_cols: List[str]) -> pd.DataFrame:
        """
        移除地理特征
        
        Args:
            geographic_cols: 地理特征列名列表
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        cols_to_drop = [col for col in geographic_cols if col in self.data.columns]
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
            logger.info(f"已移除地理特征: {cols_to_drop}")
        return self.data
        
    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            strategy: 处理策略 ('median', 'mean', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 处理数值型特征
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    median_value = self.data[col].median()
                    self.data[col].fillna(median_value, inplace=True)
                    logger.info(f"用中位数 {median_value:.2f} 填充列 {col}")
                    
        elif strategy == 'mean':
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    mean_value = self.data[col].mean()
                    self.data[col].fillna(mean_value, inplace=True)
                    
        elif strategy == 'drop':
            self.data = self.data.dropna()
            
        # 处理类别型特征
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna('None', inplace=True)
                logger.info(f"用 'None' 填充类别列 {col}")
                
        return self.data
        
    def handle_special_values(self) -> pd.DataFrame:
        """
        处理特殊值（如 '<4' 等）
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 处理 ss_effluent_mg_l 中的 '<' 符号
        if 'ss_effluent_mg_l' in self.data.columns:
            def convert_special_value(val):
                if isinstance(val, str):
                    if '<' in val:
                        # 提取数字部分
                        return float(val.replace('<', ''))
                return val
                
            self.data['ss_effluent_mg_l'] = self.data['ss_effluent_mg_l'].apply(convert_special_value)
            self.data['ss_effluent_mg_l'] = pd.to_numeric(self.data['ss_effluent_mg_l'], errors='coerce')
            logger.info("已处理 ss_effluent_mg_l 列中的特殊值")
            
        return self.data
        
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            method: 方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        initial_shape = self.data.shape[0]
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                if outliers.any():
                    logger.info(f"列 {col} 发现 {outliers.sum()} 个异常值")
                    
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.data[col]))
                outliers = z_scores > threshold
                if outliers.any():
                    logger.info(f"列 {col} 发现 {outliers.sum()} 个异常值")
                    
        final_shape = self.data.shape[0]
        logger.info(f"异常值处理完成，数据从 {initial_shape} 行变为 {final_shape} 行")
        
        return self.data
        
    def clean_data(self, 
                   drop_geographic: bool = True,
                   geographic_cols: Optional[List[str]] = None,
                   handle_missing: bool = True,
                   missing_strategy: str = 'median',
                   handle_special: bool = True,
                   remove_outliers: bool = False,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        完整的数据清洗流程
        
        Args:
            drop_geographic: 是否移除地理特征
            geographic_cols: 地理特征列名
            handle_missing: 是否处理缺失值
            missing_strategy: 缺失值处理策略
            handle_special: 是否处理特殊值
            remove_outliers: 是否移除异常值
            outlier_method: 异常值检测方法
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗...")
        
        # 1. 处理特殊值
        if handle_special:
            self.handle_special_values()
            
        # 2. 移除地理特征
        if drop_geographic and geographic_cols:
            self.drop_geographic_features(geographic_cols)
            
        # 3. 处理缺失值
        if handle_missing:
            self.handle_missing_values(missing_strategy)
            
        # 4. 移除异常值（可选）
        if remove_outliers:
            self.remove_outliers(outlier_method)
            
        logger.info(f"数据清洗完成，最终数据形状: {self.data.shape}")
        
        return self.data