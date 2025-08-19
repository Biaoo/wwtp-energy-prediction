#!/usr/bin/env python
"""
Data Analysis Script
Usage command: uv run data-analysis
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project path
sys.path.append(str(Path(__file__).parent))

from config import *
from data.data_loader import DataLoader
from data.data_cleaner import DataCleaner
from features.feature_engineering import FeatureEngineer
from visualization.visualizer import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function"""
    logger.info("=" * 50)
    logger.info("Starting data analysis...")
    logger.info("=" * 50)
    
    # 1. Load data
    logger.info("\n1. Loading data...")
    loader = DataLoader(RAW_DATA_PATH)
    data = loader.load_data()
    
    # Get data information
    data_info = loader.get_data_info()
    logger.info(f"Data shape: {data_info['shape']}")
    logger.info(f"Number of numeric columns: {len(data_info['numeric_columns'])}")
    logger.info(f"Number of categorical columns: {len(data_info['categorical_columns'])}")
    
    # Save raw data statistics
    stats_df = loader.get_basic_statistics()
    stats_path = ANALYSIS_OUTPUT_DIR / 'raw_data_statistics.csv'
    stats_df.to_csv(stats_path)
    logger.info(f"Raw data statistics saved to: {stats_path}")
    
    # 2. Data cleaning
    logger.info("\n2. Data cleaning...")
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.clean_data(
        drop_geographic=True,
        geographic_cols=GEOGRAPHIC_FEATURES,
        handle_missing=True,
        missing_strategy='median',
        handle_special=True,
        remove_outliers=False  # Don't remove outliers for now, preserve original data characteristics
    )
    
    logger.info(f"Data shape after cleaning: {cleaned_data.shape}")
    
    # 3. Feature engineering
    logger.info("\n3. Feature engineering...")
    engineer = FeatureEngineer(cleaned_data)
    
    # Create derived features
    data_with_features = engineer.create_derived_features()
    logger.info(f"Data shape after feature engineering: {data_with_features.shape}")
    
    # Save data after feature engineering
    processed_data_path = ANALYSIS_OUTPUT_DIR / 'processed_data.csv'
    data_with_features.to_csv(processed_data_path, index=False)
    logger.info(f"Processed data saved to: {processed_data_path}")
    
    # 4. Data analysis and visualization
    logger.info("\n4. Data analysis and visualization...")
    viz = Visualizer(ANALYSIS_OUTPUT_DIR)
    
    # 4.1 Data distribution analysis
    logger.info("Generating data distribution plots...")
    numeric_cols = data_with_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Select important numeric features for visualization
    important_cols = [
        TARGET_COLUMN,
        'treatment_capacity_10k_m3_per_day',
        'annual_treatment_volume_10k_m3',
        'load_rate',
        'cod_removal_rate',
        'tn_removal_rate',
        'energy_per_volume'
    ]
    
    # Filter existing columns
    cols_to_plot = [col for col in important_cols if col in numeric_cols][:12]
    if cols_to_plot:
        viz.plot_data_distribution(data_with_features, cols_to_plot)
    
    # 4.2 Correlation analysis
    logger.info("生成相关性矩阵...")
    viz.plot_correlation_matrix(data_with_features)
    
    # 4.3 Target variable correlation analysis
    if TARGET_COLUMN in data_with_features.columns:
        logger.info("分析与目标变量的相关性...")
        viz.plot_target_correlations(data_with_features, TARGET_COLUMN, top_n=20)
    
    # 5. Generate data quality report
    logger.info("\n5. 生成数据质量报告...")
    
    # 缺失值统计
    missing_stats = pd.DataFrame({
        'column': data.columns,
        'missing_count': data.isnull().sum(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).round(2)
    })
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if not missing_stats.empty:
        missing_path = ANALYSIS_OUTPUT_DIR / 'missing_values_report.csv'
        missing_stats.to_csv(missing_path, index=False)
        logger.info(f"缺失值报告已保存至: {missing_path}")
    
    # 数据类型统计
    dtype_stats = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes.astype(str),
        'unique_values': data.nunique(),
        'sample_values': [data[col].dropna().head(3).tolist() if not data[col].dropna().empty else [] for col in data.columns]
    })
    dtype_path = ANALYSIS_OUTPUT_DIR / 'data_types_report.csv'
    dtype_stats.to_csv(dtype_path, index=False)
    logger.info(f"数据类型报告已保存至: {dtype_path}")
    
    # 6. Feature analysis report
    logger.info("\n6. 生成特征分析报告...")
    
    # 数值特征统计
    numeric_features = data_with_features.select_dtypes(include=[np.number])
    feature_stats = pd.DataFrame({
        'feature': numeric_features.columns,
        'mean': numeric_features.mean(),
        'std': numeric_features.std(),
        'min': numeric_features.min(),
        'max': numeric_features.max(),
        'skewness': numeric_features.skew(),
        'kurtosis': numeric_features.kurtosis()
    })
    
    feature_stats_path = ANALYSIS_OUTPUT_DIR / 'feature_statistics.csv'
    feature_stats.to_csv(feature_stats_path, index=False)
    logger.info(f"特征统计报告已保存至: {feature_stats_path}")
    
    # 7. Treatment process analysis
    if 'treatment_process' in data.columns:
        process_stats = data.groupby('treatment_process').agg({
            TARGET_COLUMN: ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        process_path = ANALYSIS_OUTPUT_DIR / 'process_analysis.csv'
        process_stats.to_csv(process_path)
        logger.info(f"处理工艺分析已保存至: {process_path}")
    
    logger.info("\n" + "=" * 50)
    logger.info("数据分析完成！")
    logger.info(f"所有结果已保存至: {ANALYSIS_OUTPUT_DIR}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()