"""
Data Loading Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, cast

logger = logging.getLogger(__name__)

class DataLoader:
    """Data Loader"""
    
    def __init__(self, data_path: Path):
        """
        Initialize data loader
        
        Args:
            data_path: Path to data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
    def validate_columns(self, required_columns: list) -> bool:
        """
        Validate if required columns exist
        
        Args:
            required_columns: List of required column names
            
        Returns:
            bool: Whether all required columns are present
        """
        if self.data is None:
            self.load_data()
        
        self.data = cast(pd.DataFrame, self.data)
            
        missing_columns = set(required_columns) - set(self.data.columns)
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return False
        return True
        
    def get_data_info(self) -> dict:
        """
        Get basic data information
        
        Returns:
            dict: Data information
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
        Get basic statistics for numeric features
        
        Returns:
            pd.DataFrame: Statistical information
        """
        if self.data is None:
            self.load_data()
            
        self.data = cast(pd.DataFrame, self.data)
            
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.describe()