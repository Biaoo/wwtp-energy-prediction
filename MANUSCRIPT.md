# Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants

## Methods

### Data Collection and Preprocessing

We collected operational data from 93 wastewater treatment plants (WWTPs) in the Yangtze River Delta region of China, representing diverse treatment capacities ranging from 0.2 to 67.0 × 10⁴ m³/day. The dataset comprised 34 variables including influent and effluent water quality parameters (COD, BOD₅, SS, NH₃-N, TN, TP), plant capacity, annual treatment volume, treatment process types (primary, secondary, and disinfection), and annual electricity consumption as the target variable.

Data preprocessing involved systematic handling of detection limit values (e.g., "<4" for suspended solids) by converting them to half the detection limit. Missing values in categorical process features were replaced with "Unknown" category. Geographic features (city, district) were excluded to prevent overfitting and improve model generalizability. Categorical process features were one-hot encoded, expanding the feature space from 34 to 54 variables. Outlier detection using the interquartile range (IQR) method identified and retained extreme but valid observations from large-scale facilities.

### Feature Engineering

We developed domain-specific features to capture the complex relationships between treatment processes and energy consumption. Pollutant removal metrics were calculated as both absolute differences (Δ = influent - effluent) and removal rates (η = Δ/influent × 100%). Pollutant loads were computed as the product of influent concentrations and annual treatment volume (Load = C_in × V_annual), representing the total mass of pollutants processed. The plant utilization rate was derived as the ratio of actual to design capacity (λ = V_actual/(Capacity × 365)).

Process-specific features were encoded from three categorical variables: treatment process (9 categories including A2O, AO, Oxidation Ditch, SBR, MBR, Biofilm), advanced treatment (6 categories including membrane, filtration, sedimentation), and disinfection process (5 categories including chlorine-based, UV). These were transformed into 22 binary features through one-hot encoding.

An integrated efficiency metric was introduced, combining multiple removal rates: η_avg = (η_COD + η_BOD + η_SS + η_NH₃-N + η_TN + η_TP)/6. Energy intensity was calculated as kWh/m³ treated water to normalize consumption across different plant scales.

### Model Development

We implemented a comprehensive machine learning pipeline comparing seven algorithms: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest (RF), XGBoost, and LightGBM. The dataset was split into training (60%), validation (20%), and test (20%) sets using stratified sampling based on treatment capacity to ensure representative distribution across scales.

Hyperparameter optimization employed Bayesian optimization with 5-fold cross-validation. For Random Forest, we optimized n_estimators (100-500), max_depth (5-20), and min_samples_split (2-10). XGBoost parameters included learning_rate (0.01-0.1), n_estimators (100-1000), and max_depth (3-10). Feature scaling used StandardScaler for linear models while tree-based models operated on raw features to preserve interpretability.

### Model Evaluation

Model performance was assessed using multiple metrics: coefficient of determination (R²), root mean square error (RMSE), mean absolute error (MAE), and mean absolute percentage error (MAPE). Cross-validation with 5 folds ensured robust performance estimates. Feature importance was quantified using permutation importance for Random Forest and SHAP (SHapley Additive exPlanations) values for model interpretability.

Statistical significance of model differences was evaluated using paired t-tests on cross-validation scores. Residual analysis examined prediction errors across different capacity ranges to identify systematic biases.

## Results

### Data Characteristics and Correlations

The 93 WWTPs exhibited substantial heterogeneity in scale and operational parameters. Annual electricity consumption ranged from 3.58 × 10⁵ to 8.76 × 10⁷ kWh (mean ± SD: 1.31 × 10⁷ ± 1.71 × 10⁷ kWh), demonstrating high variability (CV = 131%). Treatment capacity showed similar variation with a positively skewed distribution (skewness = 2.08), indicating predominance of small to medium-scale facilities.

Correlation analysis revealed strong positive relationships between energy consumption and pollutant loads (r > 0.86 for COD, TN, and TP loads), treatment capacity (r = 0.939), and annual treatment volume (r = 0.934). Notably, pollutant loads exhibited stronger correlations than influent concentrations alone, suggesting that total mass processed drives energy demand more than concentration levels. Removal rates showed weak correlations (r < 0.3), indicating that treatment efficiency and energy consumption are not directly proportional.

### Model Performance Comparison

Among the seven models evaluated, Random Forest achieved superior performance with R² = 0.920 on the test set after incorporating process features, followed by XGBoost (R² = 0.981 on validation but with slight overfitting). Linear models (Linear Regression, Lasso) performed poorly on validation data (R² < 0.80), confirming non-linear relationships in the system. Elastic Net showed stable performance (R² = 0.941) with minimal overfitting (overfit ratio = 1.00).

The Random Forest model with expanded feature set (54 features including 22 process-related features) demonstrated robust generalization with RMSE = 5.22 × 10⁶ kWh and MAE = 2.66 × 10⁶ kWh. Notably, MAPE improved to 29.7%, a 5 percentage point reduction from the baseline model, though still exceeding typical engineering tolerance (< 20%). The inclusion of process features particularly improved predictions for medium-scale facilities while maintaining comparable performance for large-scale plants.

### Feature Importance Analysis

Feature importance analysis with the expanded feature set identified four dominant predictors accounting for 78.8% of model variance: treatment capacity (21.9%), annual treatment volume (20.7%), TN load (19.7%), and COD load (16.5%). This slight reordering from the baseline model emphasizes the primacy of scale factors, while nitrogen removal remains a critical energy driver consistent with the energy-intensive nature of nitrification-denitrification processes.

Secondary features included TP load (7.6%) and various effluent quality parameters (1-2% each). Process-related features, despite comprising 22 of the 54 total features, contributed minimally to model predictions (< 1% each), suggesting that energy consumption is primarily determined by scale and pollutant loads rather than specific treatment technologies. This finding indicates that process efficiency variations within each technology category are relatively minor compared to throughput effects.

### Model Interpretability

SHAP analysis revealed non-linear relationships between key features and energy consumption. Treatment capacity showed a threshold effect, with energy consumption increasing exponentially beyond 20 × 10⁴ m³/day. TN load exhibited a similar pattern with accelerated energy demand above 3 × 10⁶ kg/year, reflecting the disproportionate energy requirements for biological nitrogen removal at scale.

Partial dependence plots indicated interaction effects between capacity and pollutant loads. High-capacity plants treating high-strength wastewater consumed disproportionately more energy than predicted by individual feature effects, suggesting operational complexity increases with scale.

## Discussion

### Mechanistic Insights

Our findings align with established wastewater treatment principles where scale and biological nitrogen removal represent the most energy-intensive factors. After incorporating process features, the shift in importance hierarchy (capacity 21.9%, volume 20.7%, TN load 19.7%) reveals that facility scale slightly outweighs pollutant loads in determining energy consumption. This suggests that operational complexity and infrastructure requirements scale non-linearly with plant size.

The minimal importance of process type features (< 1% each) indicates remarkable convergence in energy efficiency across different treatment technologies when operated at similar scales. This suggests that modern WWTPs, regardless of specific technology (A2O, SBR, MBR), achieve comparable energy performance when treating similar loads. The weak correlation between removal rates and energy consumption further supports that WWTPs operate within narrow efficiency bands dictated by discharge standards, with energy variation primarily driven by throughput rather than treatment intensity.

### Scale Effects and Operational Implications

The exponential increase in energy consumption beyond 20 × 10⁴ m³/day capacity suggests diseconomies of scale in large WWTPs. This contradicts expected economies of scale and may reflect increased complexity in process control, longer hydraulic retention times, or more stringent effluent standards for major facilities. Despite incorporating process features, prediction errors for large plants remained high (MAPE = 29.7%), though improved from the baseline, indicating that scale-related complexity transcends simple process categorization.

Energy intensity averaging 0.44 kWh/m³ (range: 0.07-1.32) falls within global benchmarks but shows substantial optimization potential. The wide range suggests opportunities for energy efficiency improvements through operational optimization, particularly in facilities with energy intensity exceeding the 75th percentile (0.52 kWh/m³).

### Model Limitations and Future Directions

The current model's limitations include: (1) single-year snapshot data preventing seasonal variation analysis, (2) low importance of process features suggesting need for more granular process-specific parameters (SRT, HRT, MLSS), (3) lack of temporal dynamics in influent characteristics, and (4) persistent high prediction errors for large-scale facilities despite feature expansion.

The minimal contribution of process features (< 1% importance each) reveals that categorical process types alone are insufficient to capture operational nuances. Future research should incorporate continuous process parameters such as actual hydraulic retention time, sludge age, and dissolved oxygen levels. Integration of time-series data could capture seasonal patterns and dynamic loading conditions. Development of hierarchical models that first classify facilities by scale before prediction may address the heteroscedasticity observed in prediction errors.

## Conclusions

This study successfully developed a machine learning framework for predicting annual energy consumption in WWTPs, achieving R² = 0.920 using Random Forest algorithms with an expanded feature set including process characteristics. Key findings include:

1. **Scale factors are the primary energy drivers** - Treatment capacity and annual volume together explain 42.6% of energy consumption variance, surpassing even pollutant loads in importance after incorporating process features.

2. **Pollutant loads, particularly nitrogen, remain critical** - TN and COD loads account for 36.2% of model importance, emphasizing the energy-intensive nature of biological nutrient removal processes.

3. **Process types have minimal direct impact** - Despite adding 22 process-related features, their individual importance remained below 1%, suggesting that energy consumption is driven more by what is treated (scale and loads) than how it is treated (specific technologies).

4. **Prediction accuracy improved but challenges persist** - MAPE improved from 34.3% to 29.7% with feature expansion, but still exceeds engineering standards, particularly for large facilities, indicating need for more granular operational parameters.

The developed model provides a valuable tool for energy benchmarking, anomaly detection, and optimization planning in WWTPs. Implementation could support carbon footprint reduction strategies and inform design decisions for new facilities. However, achieving engineering-grade predictions (MAPE < 20%) requires incorporating temporal dynamics, process-specific parameters, and potentially physics-informed machine learning approaches that embed treatment process knowledge into model architectures.

Future deployment should focus on: (1) continuous model updating with new operational data, (2) integration with SCADA systems for real-time predictions, (3) development of uncertainty quantification methods for risk-aware decision making, and (4) extension to predict specific energy consumption components (aeration, pumping, sludge treatment) for targeted optimization.

## Supplementary Materials

### GitHub Repository

[https://github.com/Biaoo/wwtp-energy-prediction](https://github.com/Biaoo/wwtp-energy-prediction)

Citation:

```bibtex
@software{biaoo2025wwtp,
  author = {Biaoo},
  title = {Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Biaoo/wwtp-energy-prediction}},
  version = {v2.0.0}
}
```

### Data

[https://github.com/Biaoo/wwtp-energy-prediction/tree/main/data](https://github.com/Biaoo/wwtp-energy-prediction/tree/main/data)