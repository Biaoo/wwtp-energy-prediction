# Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants

## Methods

### Data Collection and Preprocessing

We collected operational data from 93 wastewater treatment plants (WWTPs) in the Yangtze River Delta region of China, representing diverse treatment capacities ranging from 0.2 to 67.0 × 10⁴ m³/day. The dataset comprised 34 variables including influent and effluent water quality parameters (COD, BOD₅, SS, NH₃-N, TN, TP), plant capacity, annual treatment volume, and annual electricity consumption as the target variable.

Data preprocessing involved systematic handling of detection limit values (e.g., "<4" for suspended solids) by converting them to half the detection limit. No missing values were observed across all features, indicating high data quality. Geographic features (city, district) were excluded to prevent overfitting and improve model generalizability. Outlier detection using the interquartile range (IQR) method identified and retained extreme but valid observations from large-scale facilities.

### Feature Engineering

We developed domain-specific features to capture the complex relationships between treatment processes and energy consumption. Pollutant removal metrics were calculated as both absolute differences (Δ = influent - effluent) and removal rates (η = Δ/influent × 100%). Pollutant loads were computed as the product of influent concentrations and annual treatment volume (Load = C_in × V_annual), representing the total mass of pollutants processed. The plant utilization rate was derived as the ratio of actual to design capacity (λ = V_actual/(Capacity × 365)).

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

Among the seven models evaluated, Random Forest achieved superior performance with R² = 0.935 on the test set, followed by XGBoost (R² = 0.981 on validation but with slight overfitting). Linear models (Linear Regression, Lasso) performed poorly on validation data (R² < 0.80), confirming non-linear relationships in the system. Elastic Net showed stable performance (R² = 0.941) with minimal overfitting (overfit ratio = 1.00).

The Random Forest model demonstrated robust generalization with RMSE = 4.68 × 10⁶ kWh and MAE = 2.61 × 10⁶ kWh. However, MAPE reached 34.3%, exceeding typical engineering tolerance (< 20%), primarily due to prediction errors in large-scale facilities. Error distribution analysis revealed Q₂₅ = 5.14 × 10⁵, Q₅₀ = 1.38 × 10⁶, and Q₇₅ = 2.52 × 10⁶ kWh, with one outlier showing residual of 1.70 × 10⁷ kWh.

### Feature Importance Analysis

Feature importance analysis identified four dominant predictors accounting for 78% of model variance: TN load (22.9%), treatment capacity (20.7%), COD load (18.3%), and annual treatment volume (16.4%). This hierarchy emphasizes the primacy of nitrogen removal in energy consumption, consistent with the energy-intensive nature of nitrification-denitrification processes.

Secondary features included TP load (6.8%) and various effluent quality parameters (1-2% each). Interestingly, removal rates contributed minimally (< 1% each), suggesting that energy consumption is driven by absolute pollutant quantities rather than removal efficiency. The average removal rate metric showed negligible importance (0.29%), further supporting this interpretation.

### Model Interpretability

SHAP analysis revealed non-linear relationships between key features and energy consumption. Treatment capacity showed a threshold effect, with energy consumption increasing exponentially beyond 20 × 10⁴ m³/day. TN load exhibited a similar pattern with accelerated energy demand above 3 × 10⁶ kg/year, reflecting the disproportionate energy requirements for biological nitrogen removal at scale.

Partial dependence plots indicated interaction effects between capacity and pollutant loads. High-capacity plants treating high-strength wastewater consumed disproportionately more energy than predicted by individual feature effects, suggesting operational complexity increases with scale.

## Discussion

### Mechanistic Insights

Our findings align with established wastewater treatment principles where biological nitrogen removal represents the most energy-intensive process. The dominance of TN load (22.9% importance) over other pollutants reflects the aerobic energy demands of nitrification (4.57 g O₂/g N) and the need for internal recirculation in denitrification. The strong correlation between pollutant loads and energy consumption (r > 0.86) validates the mass-balance approach to energy prediction.

The weak correlation between removal rates and energy consumption challenges conventional efficiency metrics. This suggests that WWTPs operate within narrow efficiency bands dictated by discharge standards, with energy variation primarily driven by throughput rather than treatment intensity. The high baseline energy consumption even at low loads indicates substantial fixed energy costs for maintaining biological processes and hydraulic operations.

### Scale Effects and Operational Implications

The exponential increase in energy consumption beyond 20 × 10⁴ m³/day capacity suggests diseconomies of scale in large WWTPs. This contradicts expected economies of scale and may reflect increased complexity in process control, longer hydraulic retention times, or more stringent effluent standards for major facilities. The higher prediction errors for large plants (MAPE = 34.3%) indicate additional unmeasured factors influencing energy consumption at scale.

Energy intensity averaging 0.44 kWh/m³ (range: 0.07-1.32) falls within global benchmarks but shows substantial optimization potential. The wide range suggests opportunities for energy efficiency improvements through operational optimization, particularly in facilities with energy intensity exceeding the 75th percentile (0.52 kWh/m³).

### Model Limitations and Future Directions

The current model's limitations include: (1) single-year snapshot data preventing seasonal variation analysis, (2) absence of process-specific parameters (SRT, HRT, MLSS), (3) lack of temporal dynamics in influent characteristics, and (4) limited representation of advanced treatment processes.

Future research should incorporate time-series data to capture seasonal patterns and dynamic loading conditions. Integration of process-specific parameters and real-time sensor data could improve prediction accuracy, particularly for large-scale facilities. Development of separate models for different capacity ranges or treatment processes may address the heteroscedasticity observed in prediction errors.

## Conclusions

This study successfully developed a machine learning framework for predicting annual energy consumption in WWTPs, achieving R² = 0.935 using Random Forest algorithms. Key findings include:

1. **Pollutant loads, particularly nitrogen, are primary energy drivers** - TN load alone explains 22.9% of energy consumption variance, emphasizing the energy-intensive nature of biological nitrogen removal.

2. **Scale and capacity dominate over efficiency metrics** - Treatment capacity and volume account for 37% of model importance, while removal rates contribute minimally, suggesting energy optimization should focus on load management rather than marginal efficiency improvements.

3. **Non-linear relationships govern energy consumption** - The superior performance of ensemble methods over linear models confirms complex interactions between operational parameters, necessitating advanced modeling approaches for accurate predictions.

4. **Prediction challenges increase with scale** - Higher errors in large facilities (MAPE = 34.3%) indicate additional complexity requiring enhanced feature sets or specialized models for different capacity ranges.

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
  version = {v1.0.0}
}
```

### Data

[https://github.com/Biaoo/wwtp-energy-prediction/tree/main/data](https://github.com/Biaoo/wwtp-energy-prediction/tree/main/data)