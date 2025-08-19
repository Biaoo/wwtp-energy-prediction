"""
特征工程模块
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化特征工程器
        
        Args:
            data: 输入数据
        """
        self.data = data.copy()
        self.scalers = {}
        self.encoders = {}
        
    def create_derived_features(self) -> pd.DataFrame:
        """
        创建衍生特征
        
        Returns:
            pd.DataFrame: 包含新特征的数据
        """
        logger.info("创建衍生特征...")
        
        # 1. 负荷率
        if 'annual_treatment_volume_10k_m3' in self.data.columns and \
           'treatment_capacity_10k_m3_per_day' in self.data.columns:
            self.data['load_rate'] = self.data['annual_treatment_volume_10k_m3'] / \
                                     (self.data['treatment_capacity_10k_m3_per_day'] * 365)
            # 处理无穷大和NaN
            self.data['load_rate'] = self.data['load_rate'].replace([np.inf, -np.inf], np.nan)
            self.data['load_rate'] = self.data['load_rate'].fillna(0)
            
        # 2. 污染物去除量
        pollutants = ['cod', 'bod5', 'ss', 'nh3n', 'tn', 'tp']
        for pollutant in pollutants:
            influent_col = f'{pollutant}_influent_mg_l'
            effluent_col = f'{pollutant}_effluent_mg_l'
            
            if influent_col in self.data.columns and effluent_col in self.data.columns:
                # 去除量
                removal_col = f'{pollutant}_removal_mg_l'
                self.data[removal_col] = self.data[influent_col] - self.data[effluent_col]
                
                # 去除率
                removal_rate_col = f'{pollutant}_removal_rate'
                self.data[removal_rate_col] = np.where(
                    self.data[influent_col] > 0,
                    (self.data[influent_col] - self.data[effluent_col]) / self.data[influent_col],
                    0
                )
                
        # 3. 污染物负荷（考虑处理量）
        if 'annual_treatment_volume_10k_m3' in self.data.columns:
            for pollutant in ['cod', 'tn', 'tp']:
                influent_col = f'{pollutant}_influent_mg_l'
                if influent_col in self.data.columns:
                    load_col = f'{pollutant}_load_kg'
                    # 转换单位：mg/L * 10k m³ * 10 = kg
                    self.data[load_col] = self.data[influent_col] * \
                                          self.data['annual_treatment_volume_10k_m3'] * 10
                                          
        # 4. 单位能耗指标
        if 'annual_electricity_consumption_kwh' in self.data.columns and \
           'annual_treatment_volume_10k_m3' in self.data.columns:
            self.data['energy_per_volume'] = np.where(
                self.data['annual_treatment_volume_10k_m3'] > 0,
                self.data['annual_electricity_consumption_kwh'] / \
                (self.data['annual_treatment_volume_10k_m3'] * 10000),  # kWh/m³
                0
            )
            
        # 5. 综合去除效率指标
        removal_rates = [col for col in self.data.columns if col.endswith('_removal_rate')]
        if removal_rates:
            self.data['avg_removal_rate'] = self.data[removal_rates].mean(axis=1)
            
        logger.info(f"创建了 {len(self.data.columns) - len(pollutants) * 4} 个衍生特征")
        
        return self.data
        
    def encode_categorical_features(self, 
                                   categorical_cols: List[str],
                                   encoding_type: str = 'onehot') -> pd.DataFrame:
        """
        编码类别特征
        
        Args:
            categorical_cols: 类别特征列名
            encoding_type: 编码类型 ('onehot', 'label')
            
        Returns:
            pd.DataFrame: 编码后的数据
        """
        logger.info(f"使用 {encoding_type} 编码类别特征...")
        
        for col in categorical_cols:
            if col not in self.data.columns:
                continue
                
            if encoding_type == 'onehot':
                # One-Hot编码
                dummies = pd.get_dummies(self.data[col], prefix=col, dummy_na=True)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data = self.data.drop(columns=[col])
                logger.info(f"对 {col} 进行One-Hot编码，生成 {len(dummies.columns)} 个特征")
                
            elif encoding_type == 'label':
                # 标签编码
                le = LabelEncoder()
                # 处理缺失值
                self.data[col] = self.data[col].fillna('None')
                self.data[col + '_encoded'] = le.fit_transform(self.data[col])
                self.encoders[col] = le
                self.data = self.data.drop(columns=[col])
                logger.info(f"对 {col} 进行标签编码")
                
        return self.data
        
    def scale_features(self, 
                      numeric_cols: List[str],
                      method: str = 'standard',
                      exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            numeric_cols: 数值特征列名
            method: 缩放方法 ('standard', 'minmax')
            exclude_cols: 不进行缩放的列
            
        Returns:
            pd.DataFrame: 缩放后的数据
        """
        if exclude_cols is None:
            exclude_cols = []
            
        cols_to_scale = [col for col in numeric_cols 
                        if col in self.data.columns and col not in exclude_cols]
        
        logger.info(f"使用 {method} 方法缩放 {len(cols_to_scale)} 个特征")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的缩放方法: {method}")
            
        if cols_to_scale:
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
            self.scalers[method] = {'scaler': scaler, 'columns': cols_to_scale}
            
        return self.data
        
    def select_features(self, 
                       target_col: str,
                       method: str = 'correlation',
                       threshold: float = 0.1) -> pd.DataFrame:
        """
        特征选择
        
        Args:
            target_col: 目标变量列名
            method: 选择方法 ('correlation', 'variance')
            threshold: 阈值
            
        Returns:
            pd.DataFrame: 选择后的特征
        """
        logger.info(f"使用 {method} 方法进行特征选择...")
        
        if method == 'correlation':
            # 基于相关性的特征选择
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            correlations = self.data[numeric_cols].corrwith(self.data[target_col]).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            
            # 确保目标变量在选择的特征中
            if target_col not in selected_features:
                selected_features.append(target_col)
                
            logger.info(f"选择了 {len(selected_features)} 个相关性大于 {threshold} 的特征")
            
        elif method == 'variance':
            # 基于方差的特征选择
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            variances = self.data[numeric_cols].var()
            selected_features = variances[variances > threshold].index.tolist()
            logger.info(f"选择了 {len(selected_features)} 个方差大于 {threshold} 的特征")
            
        else:
            selected_features = self.data.columns.tolist()
            
        return self.data[selected_features]
        
    def remove_highly_correlated_features(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        移除高度相关的特征
        
        Args:
            threshold: 相关性阈值
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        logger.info(f"移除相关性大于 {threshold} 的特征...")
        
        # 计算相关矩阵
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr().abs()
        
        # 获取上三角矩阵
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找出相关性大于阈值的特征
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            self.data = self.data.drop(columns=to_drop)
            logger.info(f"移除了 {len(to_drop)} 个高度相关的特征: {to_drop}")
            
        return self.data