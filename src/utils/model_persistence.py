"""
模型持久化模块
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
    """模型持久化管理器"""
    
    @staticmethod
    def save_model(model: Any, 
                  filepath: Path,
                  metadata: Dict = None,
                  save_format: str = 'joblib'):
        """
        保存模型
        
        Args:
            model: 要保存的模型
            filepath: 保存路径
            metadata: 模型元数据
            save_format: 保存格式 ('joblib', 'pickle')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if save_format == 'joblib':
            joblib.dump(model, filepath)
        elif save_format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"不支持的保存格式: {save_format}")
        
        logger.info(f"模型已保存至: {filepath}")
        
        # 保存元数据
        if metadata:
            metadata['save_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metadata['model_file'] = str(filepath)
            
            metadata_path = filepath.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"元数据已保存至: {metadata_path}")
    
    @staticmethod
    def load_model(filepath: Path,
                  load_format: str = 'joblib'):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            load_format: 加载格式 ('joblib', 'pickle')
            
        Returns:
            加载的模型
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        if load_format == 'joblib':
            model = joblib.load(filepath)
        elif load_format == 'pickle':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"不支持的加载格式: {load_format}")
        
        logger.info(f"模型已从 {filepath} 加载")
        
        # 尝试加载元数据
        metadata_path = filepath.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"元数据已加载: {metadata}")
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