"""
Global configuration file
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"

# Data file path
RAW_DATA_PATH = DATA_DIR / "wwtp_data_final.csv"

# Output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluation"

# Create output subdirectories
for dir_path in [ANALYSIS_OUTPUT_DIR, MODEL_OUTPUT_DIR, EVALUATION_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Random seed
RANDOM_SEED = 42

# Dataset split ratios
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Split from training set

# Feature column configuration
GEOGRAPHIC_FEATURES = ['city', 'district', 'drainage_company', 'wwtp_name']
TARGET_COLUMN = 'annual_electricity_consumption_kwh'
ID_COLUMN = 'id'

# Influent indicators
INFLUENT_FEATURES = [
    'cod_influent_mg_l', 'bod5_influent_mg_l', 'ss_influent_mg_l',
    'nh3n_influent_mg_l', 'tn_influent_mg_l', 'tp_influent_mg_l'
]

# Effluent indicators
EFFLUENT_FEATURES = [
    'cod_effluent_mg_l', 'bod5_effluent_mg_l', 'ss_effluent_mg_l',
    'nh3n_effluent_mg_l', 'tn_effluent_mg_l', 'tp_effluent_mg_l'
]

# Scale features
SCALE_FEATURES = [
    'treatment_capacity_10k_m3_per_day',
    'annual_treatment_volume_10k_m3'
]

# Process features
PROCESS_FEATURES = [
    'treatment_process',
    'advanced_processs',
    'disinfection_processs'
]

# Model hyperparameters
MODEL_PARAMS = {
    'linear_regression': {},
    'ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'max_iter': 1000
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'max_iter': 1000
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': RANDOM_SEED
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'random_state': RANDOM_SEED
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'feature_fraction': [0.7, 0.8, 0.9],
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }
}

# Cross-validation folds
CV_FOLDS = 5

# Evaluation metrics
METRICS = ['mae', 'rmse', 'r2', 'mape']