"""
数据加载模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, cast

logger = logging.getLogger(__name__)

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: Path):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"成功加载数据: {self.data.shape[0]} 行, {self.data.shape[1]} 列")
            return self.data
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
            
    def validate_columns(self, required_columns: list) -> bool:
        """
        验证必要的列是否存在
        
        Args:
            required_columns: 必需的列名列表
            
        Returns:
            bool: 是否包含所有必需列
        """
        if self.data is None:
            self.load_data()
        
        self.data = cast(pd.DataFrame, self.data)
            
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            logger.warning(f"缺少以下列: {missing_columns}")
            return False
        return True
        
    def get_data_info(self) -> dict:
        """
        获取数据基本信息
        
        Returns:
            dict: 数据信息
        """
        if self.data is None:
            self.load_data()
        
        self.data = cast(pd.DataFrame, self.data)
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        
        return info
        
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        获取数值型特征的基本统计信息
        
        Returns:
            pd.DataFrame: 统计信息
        """
        if self.data is None:
            self.load_data()
            
        self.data = cast(pd.DataFrame, self.data)
            
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.describe()