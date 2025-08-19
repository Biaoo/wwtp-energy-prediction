"""
Data cleaning module
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class DataCleaner:
    """Data cleaner"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize data cleaner
        
        Args:
            data: Raw data
        """
        self.data = data.copy()
        
    def drop_geographic_features(self, geographic_cols: List[str]) -> pd.DataFrame:
        """
        Remove geographic features
        
        Args:
            geographic_cols: List of geographic feature column names
            
        Returns:
            pd.DataFrame: Processed data
        """
        cols_to_drop = [col for col in geographic_cols if col in self.data.columns]
        if cols_to_drop:
            self.data = self.data.drop(columns=cols_to_drop)
            logger.info(f"Removed geographic features: {cols_to_drop}")
        return self.data
        
    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            strategy: Handling strategy ('median', 'mean', 'mode', 'drop')
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Handle numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if strategy == 'median':
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    median_value = self.data[col].median()
                    self.data[col].fillna(median_value, inplace=True)
                    logger.info(f"Filled column {col} with median {median_value:.2f}")
                    
        elif strategy == 'mean':
            for col in numeric_cols:
                if self.data[col].isnull().any():
                    mean_value = self.data[col].mean()
                    self.data[col].fillna(mean_value, inplace=True)
                    
        elif strategy == 'drop':
            self.data = self.data.dropna()
            
        # Handle categorical features
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna('None', inplace=True)
                logger.info(f"Filled categorical column {col} with 'None'")
                
        return self.data
        
    def handle_special_values(self) -> pd.DataFrame:
        """
        Handle special values (e.g., '<4')
        
        Returns:
            pd.DataFrame: Processed data
        """
        # Handle '<' symbol in ss_effluent_mg_l
        if 'ss_effluent_mg_l' in self.data.columns:
            def convert_special_value(val):
                if isinstance(val, str):
                    if '<' in val:
                        # Extract numeric part
                        return float(val.replace('<', ''))
                return val
                
            self.data['ss_effluent_mg_l'] = self.data['ss_effluent_mg_l'].apply(convert_special_value)
            self.data['ss_effluent_mg_l'] = pd.to_numeric(self.data['ss_effluent_mg_l'], errors='coerce')
            logger.info("Processed special values in ss_effluent_mg_l column")
            
        return self.data
        
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers
        
        Args:
            method: Method ('iqr', 'zscore')
            threshold: Threshold value
            
        Returns:
            pd.DataFrame: Processed data
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
                    logger.info(f"Found {outliers.sum()} outliers in column {col}")
                    
        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.data[col]))
                outliers = z_scores > threshold
                if outliers.any():
                    logger.info(f"Found {outliers.sum()} outliers in column {col}")
                    
        final_shape = self.data.shape[0]
        logger.info(f"Outlier removal complete, data changed from {initial_shape} to {final_shape} rows")
        
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
        Complete data cleaning pipeline
        
        Args:
            drop_geographic: Whether to remove geographic features
            geographic_cols: Geographic feature column names
            handle_missing: Whether to handle missing values
            missing_strategy: Missing value handling strategy
            handle_special: Whether to handle special values
            remove_outliers: Whether to remove outliers
            outlier_method: Outlier detection method
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning...")
        
        # 1. Handle special values
        if handle_special:
            self.handle_special_values()
            
        # 2. Remove geographic features
        if drop_geographic and geographic_cols:
            self.drop_geographic_features(geographic_cols)
            
        # 3. Handle missing values
        if handle_missing:
            self.handle_missing_values(missing_strategy)
            
        # 4. Remove outliers (optional)
        if remove_outliers:
            self.remove_outliers(outlier_method)
            
        logger.info(f"Data cleaning complete, final data shape: {self.data.shape}")
        
        return self.data