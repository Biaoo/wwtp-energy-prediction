#!/usr/bin/env python
"""
Gradio Web Interface for WWTP Energy Consumption Prediction
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import gradio as gr

# Add project path
sys.path.append(str(Path(__file__).parent))

from config import *
from features.feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Translation dictionary
TRANSLATIONS = {
    "zh": {
        "title": "污水处理厂能耗预测模型",
        "subtitle": "本系统使用机器学习模型预测污水处理厂的年度电力消耗量。\n请输入污水处理厂的运行参数进行预测。",
        "language": "语言",
        "treatment_scale": "处理规模",
        "treatment_capacity": "处理能力 (万m³/天)",
        "annual_volume": "年处理量 (万m³)",
        "process_tech": "处理工艺",
        "main_process": "主体处理工艺",
        "advanced_process": "深度处理工艺",
        "disinfection": "消毒工艺",
        "influent_quality": "进水水质 (mg/L)",
        "effluent_quality": "出水水质 (mg/L)",
        "ammonia": "氨氮",
        "total_nitrogen": "总氮",
        "total_phosphorus": "总磷",
        "predict_button": "预测能耗",
        "result_title": "### 预测结果",
        "annual_consumption": "预测的年电力消耗量",
        "unit_consumption": "单位水处理能耗",
        "input_summary": "#### 输入参数总结:",
        "treatment_scale_summary": "处理规模",
        "annual_volume_summary": "年处理量",
        "main_process_summary": "处理工艺",
        "advanced_process_summary": "深度处理工艺",
        "disinfection_summary": "消毒工艺",
        "removal_efficiency": "#### 污染物去除效率:",
        "cod_removal": "COD去除率",
        "bod5_removal": "BOD5去除率",
        "ss_removal": "SS去除率",
        "ammonia_removal": "氨氮去除率",
        "tn_removal": "总氮去除率",
        "tp_removal": "总磷去除率",
        "model_info": "#### 模型信息:",
        "model_used": "使用模型",
        "test_r2": "测试集R²",
        "error_prefix": "预测错误",
        "usage_title": "### 使用说明",
        "usage_steps": """1. 输入污水处理厂的处理规模参数
2. 选择处理工艺类型（如不清楚可选择None）
3. 输入进水和出水的水质指标
4. 点击"预测能耗"按钮获取预测结果""",
        "notes_title": "### 注意事项",
        "notes_content": """- 所有数值输入都是必需的
- 出水指标应小于或等于进水指标
- 工艺类型如不确定可选择"None"或"Other"
- 预测结果基于历史数据训练的模型，仅供参考""",
        "process_description_title": "### 工艺说明",
        "process_descriptions": """- **A2O_Family**: A2O及其变形工艺
- **AO_Family**: AO及其变形工艺
- **Oxidation_Ditch**: 氧化沟工艺
- **SBR_Family**: SBR及其变形工艺
- **MBR**: 膜生物反应器
- **Biofilm**: 生物膜工艺""",
        "none": "无",
        "per_day": "万m³/天",
        "per_year": "万m³",
        "about_title": "### 关于",
        "github_link": "GitHub仓库",
        "citation_title": "### 引用",
        "citation_content": "如果您在研究中使用了本代码，请引用：",
        "citation_text": """**Biaoo (2025)**. Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants.
GitHub repository: https://github.com/Biaoo/wwtp-energy-prediction""",
    },
    "en": {
        "title": "WWTP Energy Consumption Prediction Model",
        "subtitle": "This system uses machine learning models to predict annual electricity consumption of wastewater treatment plants.\nPlease input the operational parameters for prediction.",
        "language": "Language",
        "treatment_scale": "Treatment Scale",
        "treatment_capacity": "Treatment Capacity (10k m³/day)",
        "annual_volume": "Annual Treatment Volume (10k m³)",
        "process_tech": "Treatment Process",
        "main_process": "Main Treatment Process",
        "advanced_process": "Advanced Treatment Process",
        "disinfection": "Disinfection Process",
        "influent_quality": "Influent Quality (mg/L)",
        "effluent_quality": "Effluent Quality (mg/L)",
        "ammonia": "Ammonia Nitrogen",
        "total_nitrogen": "Total Nitrogen",
        "total_phosphorus": "Total Phosphorus",
        "predict_button": "Predict Energy",
        "result_title": "### Prediction Results",
        "annual_consumption": "Predicted Annual Electricity Consumption",
        "unit_consumption": "Unit Water Treatment Energy Consumption",
        "input_summary": "#### Input Parameters Summary:",
        "treatment_scale_summary": "Treatment Scale",
        "annual_volume_summary": "Annual Treatment Volume",
        "main_process_summary": "Treatment Process",
        "advanced_process_summary": "Advanced Treatment Process",
        "disinfection_summary": "Disinfection Process",
        "removal_efficiency": "#### Pollutant Removal Efficiency:",
        "cod_removal": "COD Removal Rate",
        "bod5_removal": "BOD5 Removal Rate",
        "ss_removal": "SS Removal Rate",
        "ammonia_removal": "Ammonia Nitrogen Removal Rate",
        "tn_removal": "Total Nitrogen Removal Rate",
        "tp_removal": "Total Phosphorus Removal Rate",
        "model_info": "#### Model Information:",
        "model_used": "Model Used",
        "test_r2": "Test Set R²",
        "error_prefix": "Prediction Error",
        "usage_title": "### Usage Instructions",
        "usage_steps": """1. Input treatment scale parameters of the WWTP
2. Select treatment process types (select None if unclear)
3. Input influent and effluent water quality indicators
4. Click "Predict Energy" button to get prediction results""",
        "notes_title": "### Notes",
        "notes_content": """- All numerical inputs are required
- Effluent indicators should be less than or equal to influent indicators
- Select "None" or "Other" if process type is uncertain
- Prediction results are based on historical data trained models, for reference only""",
        "process_description_title": "### Process Descriptions",
        "process_descriptions": """- **A2O_Family**: A2O and its variant processes
- **AO_Family**: AO and its variant processes
- **Oxidation_Ditch**: Oxidation ditch process
- **SBR_Family**: SBR and its variant processes
- **MBR**: Membrane bioreactor
- **Biofilm**: Biofilm process""",
        "none": "None",
        "per_day": "×10k m³/day",
        "per_year": "×10k m³",
        "about_title": "### About",
        "github_link": "GitHub Repository",
        "citation_title": "### Citation",
        "citation_content": "If you use this code in your research, please cite:",
        "citation_text": """**Biaoo (2025)**. Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants.
GitHub repository: https://github.com/Biaoo/wwtp-energy-prediction""",
    }
}


class WWTPPredictor:
    """WWTP Energy Consumption Predictor"""

    def __init__(self):
        """Initialize predictor"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_config = None
        self.encoders = None
        self.load_model()

    def load_model(self):
        """Load trained model and configuration"""
        try:
            # Load best model
            model_path = MODEL_OUTPUT_DIR / "best_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            # Load scaler
            scaler_path = MODEL_OUTPUT_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")

            # Load feature names
            feature_names_path = MODEL_OUTPUT_DIR / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")

            # Load model configuration
            config_path = MODEL_OUTPUT_DIR / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.model_config = json.load(f)
                logger.info(f"Model configuration loaded")

            # Load encoders if they exist
            encoder_path = MODEL_OUTPUT_DIR / "encoders.pkl"
            if encoder_path.exists():
                self.encoders = joblib.load(encoder_path)
                logger.info(f"Encoders loaded from {encoder_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, input_data):
        """Make prediction based on input features"""
        try:
            # Check if model is loaded
            if self.model is None:
                raise ValueError("Model not loaded. Please train a model first.")
            
            # Replace None values with Unknown for process features
            for col in [
                "treatment_process",
                "advanced_processs",
                "disinfection_processs",
            ]:
                if col in input_data.columns:
                    input_data[col] = input_data[col].replace("None", "Unknown")

            # Create feature engineer to calculate derived features
            engineer = FeatureEngineer(input_data)
            data_with_features = engineer.create_derived_features()

            # Encode categorical features using the same method as training
            categorical_features = [
                "treatment_process",
                "advanced_processs",
                "disinfection_processs",
            ]
            for col in categorical_features:
                if col in data_with_features.columns:
                    # Use pd.get_dummies to encode (same as training)
                    dummies = pd.get_dummies(
                        data_with_features[col], prefix=col, dummy_na=True
                    )
                    # Add encoded features to dataframe
                    for dummy_col in dummies.columns:
                        data_with_features[dummy_col] = dummies[dummy_col].values[0]
                    # Remove original categorical column
                    data_with_features = data_with_features.drop(columns=[col])

            # Ensure all required features are present
            if self.feature_names:
                # Initialize with zeros for missing features
                full_input = pd.DataFrame(0, index=[0], columns=self.feature_names)

                # Fill in provided values
                for col in data_with_features.columns:
                    if col in full_input.columns:
                        full_input[col] = data_with_features[col].values[0]

                input_data = full_input
            else:
                input_data = data_with_features

            # Apply scaling if scaler is available
            if self.scaler:
                scaled_input = self.scaler.transform(input_data)
                input_df = pd.DataFrame(scaled_input, columns=input_data.columns)
            else:
                input_df = input_data

            # Make prediction
            prediction = self.model.predict(input_df)[0]

            return prediction, data_with_features

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


def create_interface():
    """Create Gradio interface with bilingual support"""
    predictor = WWTPPredictor()

    def predict_energy(
        lang,
        treatment_capacity,
        annual_treatment_volume,
        treatment_process,
        advanced_process,
        disinfection_process,
        cod_influent,
        bod5_influent,
        ss_influent,
        nh3n_influent,
        tn_influent,
        tp_influent,
        cod_effluent,
        bod5_effluent,
        ss_effluent,
        nh3n_effluent,
        tn_effluent,
        tp_effluent,
    ):
        """Prediction function for Gradio interface"""
        try:
            # Get translations for current language
            t = TRANSLATIONS[lang]
            
            # Prepare input features as DataFrame
            input_data = pd.DataFrame(
                {
                    "treatment_capacity_10k_m3_per_day": [treatment_capacity],
                    "annual_treatment_volume_10k_m3": [annual_treatment_volume],
                    "treatment_process": [treatment_process],
                    "advanced_processs": [advanced_process],
                    "disinfection_processs": [disinfection_process],
                    "cod_influent_mg_l": [cod_influent],
                    "bod5_influent_mg_l": [bod5_influent],
                    "ss_influent_mg_l": [ss_influent],
                    "nh3n_influent_mg_l": [nh3n_influent],
                    "tn_influent_mg_l": [tn_influent],
                    "tp_influent_mg_l": [tp_influent],
                    "cod_effluent_mg_l": [cod_effluent],
                    "bod5_effluent_mg_l": [bod5_effluent],
                    "ss_effluent_mg_l": [ss_effluent],
                    "nh3n_effluent_mg_l": [nh3n_effluent],
                    "tn_effluent_mg_l": [tn_effluent],
                    "tp_effluent_mg_l": [tp_effluent],
                }
            )

            # Make prediction (this will calculate derived features internally)
            prediction, features_df = predictor.predict(input_data)

            # Calculate removal rates for display
            removal_rates = {}
            pollutants = ["cod", "bod5", "ss", "nh3n", "tn", "tp"]
            for p in pollutants:
                col_name = f"{p}_removal_rate"
                if col_name in features_df.columns:
                    removal_rates[p] = features_df[col_name].values[0] * 100
                else:
                    removal_rates[p] = 0

            # Calculate unit energy consumption
            unit_energy_consumption = prediction / (annual_treatment_volume * 10000)  # kWh/m³
            
            # Format result based on language
            result = f"""
{t['result_title']}

**{t['annual_consumption']}**: {prediction:,.2f} kWh
**{t['unit_consumption']}**: {unit_energy_consumption:.4f} kWh/m³

{t['input_summary']}
- **{t['treatment_scale_summary']}**: {treatment_capacity:.2f} {t['per_day']}
- **{t['annual_volume_summary']}**: {annual_treatment_volume:.2f} {t['per_year']}
- **{t['main_process_summary']}**: {treatment_process if treatment_process != 'None' else t['none']}
- **{t['advanced_process_summary']}**: {advanced_process if advanced_process != 'None' else t['none']}
- **{t['disinfection_summary']}**: {disinfection_process if disinfection_process != 'None' else t['none']}

{t['removal_efficiency']}
- **{t['cod_removal']}**: {removal_rates.get('cod', 0):.1f}%
- **{t['bod5_removal']}**: {removal_rates.get('bod5', 0):.1f}%
- **{t['ss_removal']}**: {removal_rates.get('ss', 0):.1f}%
- **{t['ammonia_removal']}**: {removal_rates.get('nh3n', 0):.1f}%
- **{t['tn_removal']}**: {removal_rates.get('tn', 0):.1f}%
- **{t['tp_removal']}**: {removal_rates.get('tp', 0):.1f}%

{t['model_info']}
- **{t['model_used']}**: {predictor.model_config.get('best_model', 'Unknown') if predictor.model_config else 'Unknown'}
- **{t['test_r2']}**: {predictor.model_config.get('test_metrics', {}).get('r2', 'N/A') if predictor.model_config else 'N/A'}
            """

            return result

        except Exception as e:
            t = TRANSLATIONS[lang]
            return f"{t['error_prefix']}: {str(e)}"

    def update_interface(lang):
        """Update interface text based on language selection"""
        t = TRANSLATIONS[lang]
        
        return (
            gr.update(label=t['treatment_capacity']),
            gr.update(label=t['annual_volume']),
            gr.update(label=t['main_process']),
            gr.update(label=t['advanced_process']),
            gr.update(label=t['disinfection']),
            gr.update(label=t['ammonia']),
            gr.update(label=t['total_nitrogen']),
            gr.update(label=t['total_phosphorus']),
            gr.update(label=t['ammonia']),
            gr.update(label=t['total_nitrogen']),
            gr.update(label=t['total_phosphorus']),
            gr.update(value=t['predict_button']),
            f"# {t['title']}",
            t['subtitle'],
            f"""**{t['about_title'].replace('### ', '')}**: [🔗 {t['github_link']}](https://github.com/Biaoo/wwtp-energy-prediction)""",
            f"""**{t['citation_title'].replace('### ', '')}**: Biaoo (2025). ML Framework for WWTP Energy Prediction""",
            f"### {t['treatment_scale']}",
            f"### {t['process_tech']}",
            f"### {t['influent_quality']}",
            f"### {t['effluent_quality']}",
            gr.update(value=f"""---
{t['usage_title']}
{t['usage_steps']}

{t['notes_title']}
{t['notes_content']}

{t['process_description_title']}
{t['process_descriptions']}""")
        )

    # Create Gradio interface
    with gr.Blocks(title="WWTP Energy Prediction | 污水处理厂能耗预测", theme=gr.themes.Soft()) as interface: # type: ignore
        # First row: Title and language selector
        with gr.Row():
            with gr.Column(scale=7):
                title_md = gr.Markdown(
                    """# 污水处理厂能耗预测模型"""
                )
            with gr.Column(scale=3):
                lang_selector = gr.Dropdown(
                    choices=[("中文", "zh"), ("English", "en")],
                    value="zh",
                    label="语言 / Language",
                    interactive=True,
                    elem_classes="compact-dropdown"
                )
        
        # Second row: Description, About, and Citation
        with gr.Row():
            with gr.Column(scale=4):
                desc_md = gr.Markdown(
                    """
                    本系统使用机器学习模型预测污水处理厂的年度电力消耗量。
                    请输入污水处理厂的运行参数进行预测。
                    """
                )
            with gr.Column(scale=3):
                about_md = gr.Markdown(
                    """
                    **关于**: [🔗 GitHub](https://github.com/Biaoo/wwtp-energy-prediction)
                    """
                )
            with gr.Column(scale=3):
                cite_md = gr.Markdown(
                    """
                    **引用**: Biaoo (2025). ML Framework for WWTP Energy Prediction
                    """
                )

        with gr.Row():
            with gr.Column():
                scale_md = gr.Markdown("### 处理规模")
                treatment_capacity = gr.Number(
                    label="处理能力 (万m³/天)", value=5.0, minimum=0.1, maximum=100.0
                )
                annual_treatment_volume = gr.Number(
                    label="年处理量 (万m³)", value=1500.0, minimum=10.0, maximum=50000.0
                )

                process_md = gr.Markdown("### 处理工艺")
                treatment_process = gr.Dropdown(
                    label="主体处理工艺",
                    choices=[
                        "None",
                        "A2O_Family",
                        "AO_Family",
                        "Oxidation_Ditch",
                        "SBR_Family",
                        "MBR",
                        "Biofilm",
                        "Other",
                    ],
                    value="A2O_Family",
                )
                advanced_process = gr.Dropdown(
                    label="深度处理工艺",
                    choices=[
                        "None",
                        "Membrane",
                        "Filtration",
                        "Sedimentation",
                        "Combined",
                        "Other",
                    ],
                    value="None",
                )
                disinfection_process = gr.Dropdown(
                    label="消毒工艺",
                    choices=["None", "Chlorine_Based", "UV", "Combined", "Other"],
                    value="Chlorine_Based",
                )

            with gr.Column():
                influent_md = gr.Markdown("### 进水水质 (mg/L)")
                cod_influent = gr.Number(label="COD", value=300.0, minimum=0)
                bod5_influent = gr.Number(label="BOD5", value=150.0, minimum=0)
                ss_influent = gr.Number(label="SS", value=200.0, minimum=0)
                nh3n_influent = gr.Number(label="氨氮", value=30.0, minimum=0)
                tn_influent = gr.Number(label="总氮", value=40.0, minimum=0)
                tp_influent = gr.Number(label="总磷", value=5.0, minimum=0)

            with gr.Column():
                effluent_md = gr.Markdown("### 出水水质 (mg/L)")
                cod_effluent = gr.Number(label="COD", value=30.0, minimum=0)
                bod5_effluent = gr.Number(label="BOD5", value=10.0, minimum=0)
                ss_effluent = gr.Number(label="SS", value=10.0, minimum=0)
                nh3n_effluent = gr.Number(label="氨氮", value=5.0, minimum=0)
                tn_effluent = gr.Number(label="总氮", value=15.0, minimum=0)
                tp_effluent = gr.Number(label="总磷", value=0.5, minimum=0)

        predict_btn = gr.Button("预测能耗", variant="primary")
        output = gr.Markdown()

        # Citation section at the bottom
        with gr.Accordion("📚 完整引用信息 / Full Citation", open=False):
            gr.Markdown(
                """
                ### 引用 / Citation
                
                如果您在研究中使用了本代码，请引用：
                If you use this code in your research, please cite:
                
                **Biaoo (2025)**. Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants.
                GitHub repository: https://github.com/Biaoo/wwtp-energy-prediction
                
                ```bibtex
                @software{biaoo2025wwtp,
                  author = {Biaoo},
                  title = {Machine Learning Framework for Energy Consumption Prediction in Wastewater Treatment Plants},
                  year = {2025},
                  publisher = {GitHub},
                  journal = {GitHub repository},
                  howpublished = {https://github.com/Biaoo/wwtp-energy-prediction},
                  version = {v2.0.0}
                }
                ```
                """
            )
        
        # Instructions
        instructions_md = gr.Markdown(
            """
        ---
        ### 使用说明
        1. 输入污水处理厂的处理规模参数
        2. 选择处理工艺类型（如不清楚可选择None）
        3. 输入进水和出水的水质指标
        4. 点击"预测能耗"按钮获取预测结果
        
        ### 注意事项
        - 所有数值输入都是必需的
        - 出水指标应小于或等于进水指标
        - 工艺类型如不确定可选择"None"或"Other"
        - 预测结果基于历史数据训练的模型，仅供参考
        
        ### 工艺说明
        - **A2O_Family**: A2O及其变形工艺
        - **AO_Family**: AO及其变形工艺
        - **Oxidation_Ditch**: 氧化沟工艺
        - **SBR_Family**: SBR及其变形工艺
        - **MBR**: 膜生物反应器
        - **Biofilm**: 生物膜工艺
        """
        )

        # Update interface when language changes
        lang_selector.change(
            fn=update_interface,
            inputs=[lang_selector],
            outputs=[
                treatment_capacity,
                annual_treatment_volume,
                treatment_process,
                advanced_process,
                disinfection_process,
                nh3n_influent,
                tn_influent,
                tp_influent,
                nh3n_effluent,
                tn_effluent,
                tp_effluent,
                predict_btn,
                title_md,
                desc_md,
                about_md,
                cite_md,
                scale_md,
                process_md,
                influent_md,
                effluent_md,
                instructions_md
            ]
        )

        predict_btn.click(
            fn=predict_energy,
            inputs=[
                lang_selector,
                treatment_capacity,
                annual_treatment_volume,
                treatment_process,
                advanced_process,
                disinfection_process,
                cod_influent,
                bod5_influent,
                ss_influent,
                nh3n_influent,
                tn_influent,
                tp_influent,
                cod_effluent,
                bod5_effluent,
                ss_effluent,
                nh3n_effluent,
                tn_effluent,
                tp_effluent,
            ],
            outputs=output,
        )

    return interface


import socket

def get_local_ip():
    """获取本机局域网IP地址"""
    try:
        import subprocess
        import platform
        
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'inet ' in line and '127.0.0.1' not in line and 'inet 169.254' not in line:
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        if part == 'inet' and i + 1 < len(parts):
                            ip = parts[i + 1]
                            # 检查是否是有效的局域网IP
                            if ip.startswith(('192.168.', '10.', '172.')):
                                return ip
        
        # 备用方法
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    """Main function"""
    import os
    
    try:
        logger.info("Starting Gradio interface...")
        interface = create_interface()
        
        # Get server configuration from environment or use defaults
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        
        # Check if running in Docker
        in_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER", False)
        
        if not in_docker:
            # 获取本机IP
            local_ip = get_local_ip()
            logger.info(f"Local IP: {local_ip}")
            logger.info(f"Others can access via: http://{local_ip}:{server_port}")
        
        # 启动界面
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=False,
            inbrowser=not in_docker,  # Don't open browser in Docker
        )
        
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {e}")
        raise


if __name__ == "__main__":
    main()