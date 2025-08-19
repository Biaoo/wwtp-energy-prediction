"""
Model Persistence Module
"""
import joblib
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Model Persistence Manager"""
    
    @staticmethod
    def save_model(model: Any, 
                  filepath: Path,
                  metadata: Dict = None,
                  save_format: str = 'joblib'):
        """
        Save model
        
        Args:
            model: Model to save
            filepath: Save path
            metadata: Model metadata
            save_format: Save format ('joblib', 'pickle')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if save_format == 'joblib':
            joblib.dump(model, filepath)
        elif save_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Model saved to: {filepath}")
        
        # Save metadata
        if metadata:
            metadata['save_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metadata['model_file'] = str(filepath)
            
            metadata_path = filepath.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to: {metadata_path}")
    
    @staticmethod
    def load_model(filepath: Path,
                  load_format: str = 'joblib'):
        """
        Load model
        
        Args:
            filepath: Model file path
            load_format: Load format ('joblib', 'pickle')
            
        Returns:
            Loaded model
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file does not exist: {filepath}")
        
        # Load model
        if load_format == 'joblib':
            model = joblib.load(filepath)
        elif load_format == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported load format: {load_format}")
        
        logger.info(f"Model loaded from {filepath}")
        
        # Try to load metadata
        metadata_path = filepath.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded: {metadata}")
            return model, metadata
        
        return model
    
    @staticmethod
    def save_pipeline(pipeline_dict: Dict,
                     output_dir: Path):
        """
        保存完整的机器学习pipeline
        
        Args:
            pipeline_dict: 包含各组件的字典
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各个组件
        for name, component in pipeline_dict.items():
            if component is not None:
                filepath = output_dir / f"{name}.pkl"
                joblib.dump(component, filepath)
                logger.info(f"{name} 已保存至: {filepath}")
        
        # 保存pipeline配置
        config = {
            'components': list(pipeline_dict.keys()),
            'save_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'output_dir': str(output_dir)
        }
        
        config_path = output_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Pipeline配置已保存至: {config_path}")
    
    @staticmethod
    def load_pipeline(pipeline_dir: Path) -> Dict:
        """
        加载完整的机器学习pipeline
        
        Args:
            pipeline_dir: Pipeline目录
            
        Returns:
            包含各组件的字典
        """
        pipeline_dir = Path(pipeline_dir)
        
        # 加载配置
        config_path = pipeline_dir / 'pipeline_config.json'
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载各组件
        pipeline_dict = {}
        for component_name in config['components']:
            filepath = pipeline_dir / f"{component_name}.pkl"
            if filepath.exists():
                pipeline_dict[component_name] = joblib.load(filepath)
                logger.info(f"{component_name} 已从 {filepath} 加载")
        
        return pipeline_dict