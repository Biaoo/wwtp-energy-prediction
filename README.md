# 污水处理厂能耗预测模型

基于机器学习的污水处理厂年度电力消耗预测系统。

## 项目简介

本项目使用多种机器学习算法对污水处理厂的年度能耗进行预测，基于处理规模、水质指标、处理工艺等特征构建预测模型。

## 快速开始

### 环境要求

- Python 3.10+
- UV (Python 包管理器)

### 安装

```bash
# 克隆项目
git clone https://github.com/Biaoo/wwt-prediction.git
cd wwt-prediction

# 使用UV安装依赖
uv sync
```

### 使用方法

项目提供三个主要命令：

#### 1. 数据分析

```bash
uv run python src/data_analysis.py
```

- 加载和清洗数据
- 特征工程
- 探索性数据分析（EDA）
- 生成可视化图表
- 输出位置：`data/outputs/analysis/`

#### 2. 模型训练

```bash
uv run python src/model_train.py
```

- 训练多个机器学习模型
- 自动进行超参数调优
- 模型对比和选择
- 保存最佳模型
- 输出位置：`data/outputs/models/`

#### 3. 模型评估

```bash
uv run python src/evaluate.py
```

- 加载训练好的模型
- 在测试集上评估
- 生成详细的评估报告
- 误差分析和可视化
- 输出位置：`data/outputs/evaluation/`

## 项目结构

```
wwt-prediction/
├── data/
│   ├── wwtp_data_final.csv     # 原始数据
│   └── outputs/                 # 输出目录
│       ├── analysis/            # 数据分析结果
│       ├── models/              # 训练的模型
│       └── evaluation/          # 评估结果
├── src/
│   ├── config.py                # 配置文件
│   ├── data_analysis.py         # 数据分析脚本
│   ├── model_train.py           # 模型训练脚本
│   ├── evaluate.py              # 评估脚本
│   ├── data/                    # 数据处理模块
│   ├── features/                # 特征工程模块
│   ├── models/                  # 模型训练模块
│   ├── evaluation/              # 评估模块
│   ├── visualization/           # 可视化模块
│   └── utils/                   # 工具模块
└── docs/
    └── private/
        └── init.md              # 项目设计文档
```

## 主要特性

### 数据处理

- 自动处理缺失值和异常值
- 移除地理位置特征（避免过拟合）
- 处理特殊值（如检测限）

### 特征工程

- 衍生特征：负荷率、去除率、污染物负荷等
- 类别特征编码（One-Hot、标签编码）
- 特征缩放和归一化

### 模型支持

- **基准模型**：线性回归、Ridge、Lasso、ElasticNet
- **树模型**：随机森林、XGBoost、LightGBM
- **集成方法**：模型融合和 Stacking

### 评估指标

- MAE（平均绝对误差）
- RMSE（均方根误差）
- R²（决定系数）
- MAPE（平均绝对百分比误差）

## 输出文件说明

### 数据分析输出

- `processed_data.csv` - 处理后的数据
- `correlation_matrix.png/csv` - 相关性矩阵
- `target_correlations.png/csv` - 目标变量相关性
- `data_distributions.png` - 数据分布图
- `feature_statistics.csv` - 特征统计

### 模型训练输出

- `best_model.pkl` - 最佳模型
- `scaler.pkl` - 特征缩放器
- `model_config.json` - 模型配置
- `training_results.json` - 训练结果
- `model_comparison.png/csv` - 模型对比

### 评估输出

- `evaluation_report.md` - Markdown 格式报告
- `final_evaluation_report.json` - 详细评估结果
- `error_analysis.json` - 误差分析
- `segment_evaluation.csv` - 分段评估

## 性能指标

目标性能：

- R² > 0.85
- MAPE < 15%

## 注意事项

1. 数据集规模较小（93 条），建议使用交叉验证
2. macOS 用户需要安装 OpenMP 支持 XGBoost
3. 所有输出都包含对应的 CSV 文件，便于二次分析

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

MIT License
