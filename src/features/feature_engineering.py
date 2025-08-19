"""
Feature Engineering Module
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature Engineer"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            data: Input data
        """
        self.data = data.copy()
        self.scalers = {}
        self.encoders = {}
        
    def create_derived_features(self) -> pd.DataFrame:
        """
        Create derived features
        
        Returns:
            pd.DataFrame: Data with new features
        """
        logger.info("Creating derived features...")
        
        # 1. Load rate
        if 'annual_treatment_volume_10k_m3' in self.data.columns and \
           'treatment_capacity_10k_m3_per_day' in self.data.columns:
            self.data['load_rate'] = self.data['annual_treatment_volume_10k_m3'] / \
                                     (self.data['treatment_capacity_10k_m3_per_day'] * 365)
            # Handle infinity and NaN values
            self.data['load_rate'] = self.data['load_rate'].replace([np.inf, -np.inf], np.nan)
            self.data['load_rate'] = self.data['load_rate'].fillna(0)
            
        # 2. Pollutant removal amounts
        pollutants = ['cod', 'bod5', 'ss', 'nh3n', 'tn', 'tp']
        for pollutant in pollutants:
            influent_col = f'{pollutant}_influent_mg_l'
            effluent_col = f'{pollutant}_effluent_mg_l'
            
            if influent_col in self.data.columns and effluent_col in self.data.columns:
                # Removal amount
                removal_col = f'{pollutant}_removal_mg_l'
                self.data[removal_col] = self.data[influent_col] - self.data[effluent_col]
                
                # Removal rate
                removal_rate_col = f'{pollutant}_removal_rate'
                self.data[removal_rate_col] = np.where(
                    self.data[influent_col] > 0,
                    (self.data[influent_col] - self.data[effluent_col]) / self.data[influent_col],
                    0
                )
                
        # 3. Pollutant loads (considering treatment volume)
        if 'annual_treatment_volume_10k_m3' in self.data.columns:
            for pollutant in ['cod', 'tn', 'tp']:
                influent_col = f'{pollutant}_influent_mg_l'
                if influent_col in self.data.columns:
                    load_col = f'{pollutant}_load_kg'
                    # Unit conversion: mg/L * 10k m³ * 10 = kg
                    self.data[load_col] = self.data[influent_col] * \
                                          self.data['annual_treatment_volume_10k_m3'] * 10
                                          
        # 4. Unit energy consumption indicators
        if 'annual_electricity_consumption_kwh' in self.data.columns and \
           'annual_treatment_volume_10k_m3' in self.data.columns:
            self.data['energy_per_volume'] = np.where(
                self.data['annual_treatment_volume_10k_m3'] > 0,
                self.data['annual_electricity_consumption_kwh'] / \
                (self.data['annual_treatment_volume_10k_m3'] * 10000),  # kWh/m³
                0
            )
            
        # 5. Comprehensive removal efficiency indicators
        removal_rates = [col for col in self.data.columns if col.endswith('_removal_rate')]
        if removal_rates:
            self.data['avg_removal_rate'] = self.data[removal_rates].mean(axis=1)
            
        logger.info(f"Created {len(self.data.columns) - len(pollutants) * 4} derived features")
        
        return self.data
        
    def encode_categorical_features(self, 
                                   categorical_cols: List[str],
                                   encoding_type: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            categorical_cols: Categorical feature column names
            encoding_type: Encoding type ('onehot', 'label')
            
        Returns:
            pd.DataFrame: Encoded data
        """
        logger.info(f"Encoding categorical features using {encoding_type}...")
        
        for col in categorical_cols:
            if col not in self.data.columns:
                continue
                
            if encoding_type == 'onehot':
                # One-Hot encoding
                dummies = pd.get_dummies(self.data[col], prefix=col, dummy_na=True)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data = self.data.drop(columns=[col])
                logger.info(f"One-Hot encoded {col}, generated {len(dummies.columns)} features")
                
            elif encoding_type == 'label':
                # Label encoding
                le = LabelEncoder()
                # Handle missing values
                self.data[col] = self.data[col].fillna('None')
                self.data[col + '_encoded'] = le.fit_transform(self.data[col])
                self.encoders[col] = le
                self.data = self.data.drop(columns=[col])
                logger.info(f"Label encoded {col}")
                
        return self.data
        
    def scale_features(self, 
                      numeric_cols: List[str],
                      method: str = 'standard',
                      exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Feature scaling
        
        Args:
            numeric_cols: Numeric feature column names
            method: Scaling method ('standard', 'minmax')
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            pd.DataFrame: Scaled data
        """
        if exclude_cols is None:
            exclude_cols = []
            
        cols_to_scale = [col for col in numeric_cols 
                        if col in self.data.columns and col not in exclude_cols]
        
        logger.info(f"Scaling {len(cols_to_scale)} features using {method} method")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
        if cols_to_scale:
            self.data[cols_to_scale] = scaler.fit_transform(self.data[cols_to_scale])
            self.scalers[method] = {'scaler': scaler, 'columns': cols_to_scale}
            
        return self.data
        
    def select_features(self, 
                       target_col: str,
                       method: str = 'correlation',
                       threshold: float = 0.1) -> pd.DataFrame:
        """
        Feature selection
        
        Args:
            target_col: Target variable column name
            method: Selection method ('correlation', 'variance')
            threshold: Threshold value
            
        Returns:
            pd.DataFrame: Selected features
        """
        logger.info(f"Performing feature selection using {method} method...")
        
        if method == 'correlation':
            # Correlation-based feature selection
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            correlations = self.data[numeric_cols].corrwith(self.data[target_col]).abs()
            selected_features = correlations[correlations > threshold].index.tolist()
            
            # Ensure target variable is in selected features
            if target_col not in selected_features:
                selected_features.append(target_col)
                
            logger.info(f"Selected {len(selected_features)} features with correlation > {threshold}")
            
        elif method == 'variance':
            # Variance-based feature selection
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            variances = self.data[numeric_cols].var()
            selected_features = variances[variances > threshold].index.tolist()
            logger.info(f"Selected {len(selected_features)} features with variance > {threshold}")
            
        else:
            selected_features = self.data.columns.tolist()
            
        return self.data[selected_features]
        
    def remove_highly_correlated_features(self, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features
        
        Args:
            threshold: Correlation threshold
            
        Returns:
            pd.DataFrame: Processed data
        """
        logger.info(f"Removing features with correlation > {threshold}...")
        
        # Calculate correlation matrix
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr().abs()
        
        # Get upper triangle matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation above threshold
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            self.data = self.data.drop(columns=to_drop)
            logger.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
            
        return self.data