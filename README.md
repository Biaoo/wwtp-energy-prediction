# Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1234%2Fexample-blue)](https://doi.org/10.1234/example)
[![Dataset](https://img.shields.io/badge/Dataset-Available-brightgreen)](data/)
[![Models](https://img.shields.io/badge/Models-Pretrained-orange)](data/outputs/models/)

## Overview

This repository provides a comprehensive machine learning framework for predicting energy consumption in wastewater treatment plants (WWTPs). The framework implements and compares seven state-of-the-art algorithms, achieving high-precision predictions that support environmental impact assessments and operational optimization.

This codebase serves as the computational foundation for the research paper: **"AI-Driven Wastewater Treatment LCA Dataset: High Precision, Dynamic Updates, and Regional Applicability Study"**, providing all data processing, model training, and evaluation pipelines used in the study.

## Key Features

### 1. High-Precision AI Models

- **Random Forest Performance**: R² = 0.935, RMSE = 4.68 × 10⁶ kWh
- **Feature Importance**: TN load (22.9%), treatment capacity (20.7%), COD load (18.3%)
- **Energy Intensity Prediction**: 0.44 ± 0.19 kWh/m³ (regional average)

### 2. Comprehensive Dataset

- **Scale**: 93 WWTPs from the Yangtze River Delta region
- **Coverage**: Treatment capacities from 0.2 to 67.0 × 10⁴ m³/day
- **Features**: 34 operational variables including water quality parameters and plant metrics

### 3. Complete ML Pipeline

- **Data Processing**: Automated cleaning, outlier detection, and feature engineering
- **Model Selection**: Seven algorithms with hyperparameter optimization
- **Evaluation**: Multiple metrics (R², RMSE, MAE, MAPE) with cross-validation
- **Visualization**: Publication-ready figures and performance comparisons

## Citation

If you use this code in your research, please cite both the paper and this GitHub repository:

### Citing the Code

```bibtex
@software{biaoo2025wwtp,
  author = {Biaoo},
  title = {Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Biaoo/wwtp-energy-prediction}},
  version = {v1.0.0}
}
```

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager ([installation guide](https://github.com/astral-sh/uv))

### Setup

```bash
# Clone the repository
git clone https://github.com/Biaoo/wwtp-energy-prediction.git
cd wwtp-energy-prediction

# Install dependencies using UV
uv sync

# Verify installation
uv run python --version
```

## Reproducibility

To reproduce the results from the paper, execute the following commands in sequence:

### 1. Data Analysis and Feature Engineering

```bash
uv run python src/data_analysis.py
```

This script performs:

- Data cleaning and preprocessing (n=93 WWTPs)
- Feature engineering (34 derived features)
- Exploratory data analysis with correlation matrices
- Statistical summaries and distribution plots

**Output**: `data/outputs/analysis/` containing:

- `correlation_matrix.csv/png`: Feature correlation analysis
- `feature_statistics.csv`: Comprehensive statistical metrics
- `processed_data.csv`: Clean dataset with engineered features

### 2. Model Training and Optimization

```bash
uv run python src/model_train.py
```

This script implements:

- Seven ML algorithms (Linear, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM)
- Bayesian hyperparameter optimization
- 5-fold cross-validation
- Feature importance analysis using SHAP

**Output**: `data/outputs/models/` containing:

- `best_model.pkl`: Optimized Random Forest model
- `model_comparison.csv`: Performance metrics for all models
- `feature_importance.csv/png`: Feature ranking and visualization

### 3. Model Evaluation and Validation

```bash
uv run python src/evaluate.py
```

This script provides:

- Test set evaluation (n=19, 20% of data)
- Residual analysis and error distribution
- Performance metrics (R², RMSE, MAE, MAPE)
- Prediction visualizations

**Output**: `data/outputs/evaluation/` containing:

- `evaluation_report.md`: Comprehensive performance report
- `predictions.csv`: Actual vs. predicted values
- `residual_plots.png`: Error analysis visualizations

## Data Description

### Dataset Overview

- **Sample Size**: 93 wastewater treatment plants
- **Region**: Yangtze River Delta, China
- **Time Period**: Annual averages (2022)
- **Features**: 34 variables including water quality parameters, plant capacity, and operational metrics

### Key Variables

| Variable Category | Description                      | Example Features                                 |
| ----------------- | -------------------------------- | ------------------------------------------------ |
| Water Quality     | Influent/effluent concentrations | COD, BOD₅, SS, NH₃-N, TN, TP                     |
| Plant Capacity    | Treatment scale metrics          | Design capacity, annual volume                   |
| Derived Features  | Engineered variables             | Pollutant loads, removal rates, energy intensity |
| Target Variable   | Energy consumption               | Annual electricity (kWh)                         |

## Repository Structure

```
wwtp-energy-prediction/
│
├── data/
│   ├── wwtp_data_final.csv          # Original dataset (93 WWTPs)
│   └── outputs/
│       ├── analysis/                 # EDA results and visualizations
│       │   ├── correlation_matrix.*  # Feature correlations
│       │   ├── feature_statistics.csv
│       │   └── processed_data.csv
│       ├── models/                   # Trained models and comparisons
│       │   ├── best_model.pkl        # Random Forest (R²=0.935)
│       │   ├── model_comparison.csv
│       │   └── feature_importance.*
│       └── evaluation/               # Performance metrics
│           ├── evaluation_report.md
│           ├── predictions.csv
│           └── residual_plots.png
│
├── src/
│   ├── config.py                    # Global configuration
│   ├── data_analysis.py             # EDA and preprocessing
│   ├── model_train.py                # Model training pipeline
│   ├── evaluate.py                   # Model evaluation
│   │
│   ├── data/                         # Data processing modules
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── data_splitter.py
│   │
│   ├── features/                     # Feature engineering
│   │   ├── feature_engineering.py
│   │   └── feature_selector.py
│   │
│   ├── models/                       # ML model implementations
│   │   ├── baseline_models.py
│   │   ├── tree_models.py
│   │   └── model_selector.py
│   │
│   └── visualization/                # Plotting utilities
│       ├── plot_utils.py
│       └── evaluation_plots.py
│
├── docs/
│   └── private/
│       ├── init.md                   # Research design
│       ├── summary.md                # Results summary
│       └── method&discussion.md      # Paper draft
│
├── uv.lock                           # Dependency lock file
├── pyproject.toml                    # Project configuration
└── README.md                         # This file
```

## Technical Implementation

### Feature Engineering

- **Pollutant Loads**: Load = Concentration × Annual Volume
- **Removal Rates**: η = (Influent - Effluent) / Influent × 100%
- **Load Rate**: λ = Actual Volume / (Design Capacity × 365)
- **Energy Intensity**: kWh/m³ treated water

### Model Architecture

| Model         | Validation R² | Test R² | MAPE (%) | Status         |
| ------------- | ------------- | ------- | -------- | -------------- |
| Random Forest | 0.985         | 0.935   | 34.3     | **Selected**   |
| XGBoost       | 0.981         | -       | -        | Slight overfit |
| Elastic Net   | 0.941         | -       | -        | Stable         |
| Ridge         | 0.913         | -       | -        | Baseline       |
| Linear/Lasso  | <0.80         | -       | -        | Underfit       |

### Performance Metrics

- **R² Score**: 0.935 (excellent fit)
- **RMSE**: 4.68 × 10⁶ kWh
- **MAE**: 2.61 × 10⁶ kWh
- **MAPE**: 34.3% (higher for large facilities)

## Requirements

### Software Dependencies

```
"lightgbm>=4.6.0",
"matplotlib>=3.10.5",
"numpy>=2.2.6",
"openpyxl>=3.1.5",
"pandas>=2.3.1",
"scikit-learn>=1.7.1",
"xgboost>=3.0.4",
"seaborn>=0.13.0",
"joblib>=1.3.0",
"statsmodels>=0.14.5",
```

### Hardware Requirements

- **Minimum**: 8GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 4+ CPU cores
- **Storage**: ~500MB for code and outputs

## Contributing

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Reporting Issues

Please use the [GitHub Issues](https://github.com/Biaoo/wwtp-energy-prediction/issues) page to report bugs or request features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or collaboration opportunities:

- **Author**: [Biaoo](biao00luo@gmail.com)
- **Project Homepage**: [https://github.com/Biaoo/wwtp-energy-prediction](https://github.com/Biaoo/wwtp-energy-prediction)

---

_Last updated: August 2025_
