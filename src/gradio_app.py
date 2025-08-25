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
    """Create Gradio interface"""
    predictor = WWTPPredictor()

    def predict_energy(
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
            
            # Format result
            result = f"""
### 预测结果

**预测的年电力消耗量**: {prediction:,.2f} kWh
**单位水处理能耗**: {unit_energy_consumption:.4f} kWh/m³

#### 输入参数总结:
- **处理规模**: {treatment_capacity:.2f} 万m³/天
- **年处理量**: {annual_treatment_volume:.2f} 万m³
- **处理工艺**: {treatment_process if treatment_process != 'None' else '无'}
- **深度处理工艺**: {advanced_process if advanced_process != 'None' else '无'}
- **消毒工艺**: {disinfection_process if disinfection_process != 'None' else '无'}

#### 污染物去除效率:
- **COD去除率**: {removal_rates.get('cod', 0):.1f}%
- **BOD5去除率**: {removal_rates.get('bod5', 0):.1f}%
- **SS去除率**: {removal_rates.get('ss', 0):.1f}%
- **氨氮去除率**: {removal_rates.get('nh3n', 0):.1f}%
- **总氮去除率**: {removal_rates.get('tn', 0):.1f}%
- **总磷去除率**: {removal_rates.get('tp', 0):.1f}%

#### 模型信息:
- **使用模型**: {predictor.model_config.get('best_model', 'Unknown') if predictor.model_config else 'Unknown'}
- **测试集R²**: {predictor.model_config.get('test_metrics', {}).get('r2', 'N/A') if predictor.model_config else 'N/A'}
            """

            return result

        except Exception as e:
            return f"预测错误: {str(e)}"

    # Create Gradio interface
    with gr.Blocks(title="污水处理厂能耗预测系统") as interface:
        gr.Markdown(
            """
        # 污水处理厂能耗预测系统
        
        本系统使用机器学习模型预测污水处理厂的年度电力消耗量。
        请输入污水处理厂的运行参数进行预测。
        
        """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 处理规模")
                treatment_capacity = gr.Number(
                    label="处理能力 (万m³/天)", value=5.0, minimum=0.1, maximum=100.0
                )
                annual_treatment_volume = gr.Number(
                    label="年处理量 (万m³)", value=1500.0, minimum=10.0, maximum=50000.0
                )

                gr.Markdown("### 处理工艺")
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
                gr.Markdown("### 进水水质 (mg/L)")
                cod_influent = gr.Number(label="COD", value=300.0, minimum=0)
                bod5_influent = gr.Number(label="BOD5", value=150.0, minimum=0)
                ss_influent = gr.Number(label="SS", value=200.0, minimum=0)
                nh3n_influent = gr.Number(label="氨氮", value=30.0, minimum=0)
                tn_influent = gr.Number(label="总氮", value=40.0, minimum=0)
                tp_influent = gr.Number(label="总磷", value=5.0, minimum=0)

            with gr.Column():
                gr.Markdown("### 出水水质 (mg/L)")
                cod_effluent = gr.Number(label="COD", value=30.0, minimum=0)
                bod5_effluent = gr.Number(label="BOD5", value=10.0, minimum=0)
                ss_effluent = gr.Number(label="SS", value=10.0, minimum=0)
                nh3n_effluent = gr.Number(label="氨氮", value=5.0, minimum=0)
                tn_effluent = gr.Number(label="总氮", value=15.0, minimum=0)
                tp_effluent = gr.Number(label="总磷", value=0.5, minimum=0)

        predict_btn = gr.Button("预测能耗", variant="primary")
        output = gr.Markdown()

        predict_btn.click(
            fn=predict_energy,
            inputs=[
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

        gr.Markdown(
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
    try:
        logger.info("Starting Gradio interface...")
        interface = create_interface()
        
        # 获取本机IP
        local_ip = get_local_ip()
        logger.info(f"Local IP: {local_ip}")
        port = 7866
        logger.info(f"Others can access via: http://{local_ip}:{port}")
        
        # 启动界面
        interface.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=port,
            share=False,
            inbrowser=True,
        )
        
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {e}")
        raise


if __name__ == "__main__":
    main()
