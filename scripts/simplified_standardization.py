import pandas as pd
from enum import Enum
from typing import Dict

class SimplifiedTreatmentProcess(Enum):
    """简化的主处理工艺枚举 - 基于技术相似性合并"""
    A2O_FAMILY = "A2O_Family"  # A2O、Modified_A2O、Multistage_A2O 合并
    AO_FAMILY = "AO_Family"  # AO、Modified_AO 合并
    OXIDATION_DITCH = "Oxidation_Ditch"  # 所有氧化沟类型合并
    SBR_FAMILY = "SBR_Family"  # SBR、MSBR、CASS、CAST 合并
    MBR = "MBR"  # 膜生物反应器保持独立
    BIOFILM = "Biofilm"  # MBBR、BAF、Contact_Oxidation 等生物膜法合并
    OTHER = "Other"
    NONE = "None"

class SimplifiedAdvancedTreatment(Enum):
    """简化的深度处理工艺枚举"""
    FILTRATION = "Filtration"  # 所有滤池类型合并
    SEDIMENTATION = "Sedimentation"  # 高效沉淀类
    MEMBRANE = "Membrane"  # 所有膜处理合并
    COMBINED = "Combined"  # 组合工艺
    OTHER = "Other"
    NONE = "None"

class SimplifiedDisinfection(Enum):
    """简化的消毒方式枚举"""
    CHLORINE_BASED = "Chlorine_Based"  # 所有氯消毒合并
    UV = "UV"  # 紫外线
    COMBINED = "Combined"  # 组合消毒
    OTHER = "Other"
    NONE = "None"

def apply_simplified_standardization():
    # 读取之前标准化的数据
    df = pd.read_csv('data/wwtp_data_standardized.csv')
    
    print("=" * 80)
    print("污水处理工艺简化标准化")
    print("=" * 80)
    
    # 1. 创建简化的主处理工艺映射
    treatment_simplification = {
        'A2O': SimplifiedTreatmentProcess.A2O_FAMILY.value,
        'Modified_A2O': SimplifiedTreatmentProcess.A2O_FAMILY.value,
        'Multistage_A2O': SimplifiedTreatmentProcess.A2O_FAMILY.value,
        
        'AO': SimplifiedTreatmentProcess.AO_FAMILY.value,
        'Modified_AO': SimplifiedTreatmentProcess.AO_FAMILY.value,
        
        'Oxidation_Ditch': SimplifiedTreatmentProcess.OXIDATION_DITCH.value,
        'Modified_Oxidation_Ditch': SimplifiedTreatmentProcess.OXIDATION_DITCH.value,
        
        'SBR': SimplifiedTreatmentProcess.SBR_FAMILY.value,
        'MSBR': SimplifiedTreatmentProcess.SBR_FAMILY.value,
        'CASS': SimplifiedTreatmentProcess.SBR_FAMILY.value,
        'CAST': SimplifiedTreatmentProcess.SBR_FAMILY.value,
        
        'MBR': SimplifiedTreatmentProcess.MBR.value,
        
        'MBBR': SimplifiedTreatmentProcess.BIOFILM.value,
        'BAF': SimplifiedTreatmentProcess.BIOFILM.value,
        'Contact_Oxidation': SimplifiedTreatmentProcess.BIOFILM.value,
        'Activated_Sludge': SimplifiedTreatmentProcess.OTHER.value,
        
        'UNITANK': SimplifiedTreatmentProcess.OTHER.value,
        'Other': SimplifiedTreatmentProcess.OTHER.value,
        'None': SimplifiedTreatmentProcess.NONE.value,
    }
    
    # 2. 创建简化的深度处理映射
    advanced_simplification = {
        'V_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'Cloth_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'Disk_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'Sand_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'Denitrification_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'Deep_Bed_Filter': SimplifiedAdvancedTreatment.FILTRATION.value,
        'BAF': SimplifiedAdvancedTreatment.FILTRATION.value,
        
        'High_Efficiency_Sedimentation': SimplifiedAdvancedTreatment.SEDIMENTATION.value,
        
        'Membrane_Filter': SimplifiedAdvancedTreatment.MEMBRANE.value,
        'MBR_Membrane': SimplifiedAdvancedTreatment.MEMBRANE.value,
        
        'Combined_Process': SimplifiedAdvancedTreatment.COMBINED.value,
        
        'Ozone': SimplifiedAdvancedTreatment.OTHER.value,
        'Other': SimplifiedAdvancedTreatment.OTHER.value,
        'None': SimplifiedAdvancedTreatment.NONE.value,
    }
    
    # 3. 创建简化的消毒方式映射
    disinfection_simplification = {
        'Sodium_Hypochlorite': SimplifiedDisinfection.CHLORINE_BASED.value,
        'Chlorine': SimplifiedDisinfection.CHLORINE_BASED.value,
        'Chlorine_Dioxide': SimplifiedDisinfection.CHLORINE_BASED.value,
        
        'UV': SimplifiedDisinfection.UV.value,
        
        'UV_Plus_Sodium_Hypochlorite': SimplifiedDisinfection.COMBINED.value,
        
        'Ozone': SimplifiedDisinfection.OTHER.value,
        'Other': SimplifiedDisinfection.OTHER.value,
        'None': SimplifiedDisinfection.NONE.value,
    }
    
    # 应用简化映射
    df['treatment_simplified'] = df['treatment_process_std'].map(treatment_simplification)
    df['advanced_simplified'] = df['advanced_treatment_std'].map(advanced_simplification)
    df['disinfection_simplified'] = df['disinfection_method_std'].map(disinfection_simplification)
    
    # 统计分析
    print("\n原始标准化 vs 简化标准化对比")
    print("-" * 60)
    
    print("\n1. 主处理工艺简化")
    print("原标准化类型数:", df['treatment_process_std'].nunique())
    print("简化后类型数:", df['treatment_simplified'].nunique())
    
    print("\n简化后分布:")
    treatment_counts = df['treatment_simplified'].value_counts()
    for process, count in treatment_counts.items():
        print(f"  {process}: {count}家 ({count/len(df)*100:.1f}%)")
    
    print("\n2. 深度处理工艺简化")
    print("原标准化类型数:", df['advanced_treatment_std'].nunique())
    print("简化后类型数:", df['advanced_simplified'].nunique())
    
    print("\n简化后分布:")
    advanced_counts = df['advanced_simplified'].value_counts()
    for process, count in advanced_counts.items():
        print(f"  {process}: {count}家 ({count/len(df)*100:.1f}%)")
    
    print("\n3. 消毒方式简化")
    print("原标准化类型数:", df['disinfection_method_std'].nunique())
    print("简化后类型数:", df['disinfection_simplified'].nunique())
    
    print("\n简化后分布:")
    disinfection_counts = df['disinfection_simplified'].value_counts()
    for method, count in disinfection_counts.items():
        print(f"  {method}: {count}家 ({count/len(df)*100:.1f}%)")
    
    # 创建工艺组合分析
    print("\n" + "=" * 80)
    print("工艺组合分析")
    print("=" * 80)
    
    # 统计主处理-深度处理组合
    print("\n最常见的工艺组合 (主处理 + 深度处理):")
    combo_counts = df.groupby(['treatment_simplified', 'advanced_simplified']).size().sort_values(ascending=False)
    for (treatment, advanced), count in combo_counts.head(10).items():
        print(f"  {treatment} + {advanced}: {count}家 ({count/len(df)*100:.1f}%)")
    
    # 统计完整组合
    print("\n最常见的完整工艺组合 (主处理 + 深度处理 + 消毒):")
    full_combo = df.groupby(['treatment_simplified', 'advanced_simplified', 'disinfection_simplified']).size().sort_values(ascending=False)
    for (treatment, advanced, disinfection), count in full_combo.head(10).items():
        print(f"  {treatment} + {advanced} + {disinfection}: {count}家 ({count/len(df)*100:.1f}%)")
    
    # 保存结果
    output_file = 'data/wwtp_data_simplified.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n简化标准化数据已保存至: {output_file}")
    
    # 输出简化映射关系说明
    print("\n" + "=" * 80)
    print("简化映射说明")
    print("=" * 80)
    
    print("\n主处理工艺合并说明:")
    print("  • A2O_Family: 包含所有A2O变体（标准、改良、多级等）")
    print("  • AO_Family: 包含AO及其改良型")
    print("  • Oxidation_Ditch: 包含所有氧化沟类型")
    print("  • SBR_Family: 包含SBR、MSBR、CASS、CAST等序批式反应器")
    print("  • MBR: 膜生物反应器（独立类别，技术特征独特）")
    print("  • Biofilm: 生物膜法（MBBR、BAF、接触氧化等）")
    
    print("\n深度处理工艺合并说明:")
    print("  • Filtration: 所有滤池类型（V型、滤布、转盘、砂滤、反硝化等）")
    print("  • Sedimentation: 高效沉淀池类")
    print("  • Membrane: 膜处理技术")
    print("  • Combined: 组合工艺（沉淀+过滤等）")
    
    print("\n消毒方式合并说明:")
    print("  • Chlorine_Based: 所有氯系消毒（次氯酸钠、液氯、二氧化氯）")
    print("  • UV: 紫外线消毒")
    print("  • Combined: 组合消毒（UV+氯等）")
    
    return df

def create_process_features(df):
    """基于简化的工艺类型创建特征工程"""
    print("\n" + "=" * 80)
    print("工艺特征工程")
    print("=" * 80)
    
    # 创建工艺复杂度评分
    complexity_scores = {
        'A2O_Family': 3,  # 生物脱氮除磷，复杂度高
        'MBR': 4,  # 膜生物反应器，技术复杂
        'SBR_Family': 2,  # 序批式，操作相对简单
        'Oxidation_Ditch': 2,  # 氧化沟，运行稳定
        'AO_Family': 2,  # 缺氧好氧，中等复杂度
        'Biofilm': 2,  # 生物膜法
        'Other': 1,
        'None': 0
    }
    
    df['treatment_complexity'] = df['treatment_simplified'].map(complexity_scores).fillna(1)
    
    # 创建是否有深度处理标记
    df['has_advanced_treatment'] = (df['advanced_simplified'] != 'None').astype(int)
    
    # 创建是否为组合消毒标记
    df['has_combined_disinfection'] = (df['disinfection_simplified'] == 'Combined').astype(int)
    
    # 创建总体工艺复杂度评分
    advanced_scores = {
        'Membrane': 3,
        'Combined': 2,
        'Filtration': 2,
        'Sedimentation': 1,
        'Other': 1,
        'None': 0
    }
    
    df['advanced_complexity'] = df['advanced_simplified'].map(advanced_scores).fillna(0)
    df['total_complexity'] = df['treatment_complexity'] + df['advanced_complexity']
    
    print("\n工艺复杂度统计:")
    print(f"平均处理工艺复杂度: {df['treatment_complexity'].mean():.2f}")
    print(f"平均深度处理复杂度: {df['advanced_complexity'].mean():.2f}")
    print(f"平均总体复杂度: {df['total_complexity'].mean():.2f}")
    
    print(f"\n有深度处理的污水厂: {df['has_advanced_treatment'].sum()}家 ({df['has_advanced_treatment'].mean()*100:.1f}%)")
    print(f"采用组合消毒的污水厂: {df['has_combined_disinfection'].sum()}家 ({df['has_combined_disinfection'].mean()*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    df = apply_simplified_standardization()
    df = create_process_features(df)
    
    # 保存最终版本
    df.to_csv('data/wwtp_data_final.csv', index=False, encoding='utf-8-sig')
    print(f"\n最终版本数据已保存至: data/wwtp_data_final.csv")