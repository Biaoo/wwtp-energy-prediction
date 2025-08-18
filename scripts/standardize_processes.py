import pandas as pd
from enum import Enum
from typing import Dict, Optional

class TreatmentProcess(Enum):
    """主处理工艺标准枚举"""
    A2O = "A2O"
    MODIFIED_A2O = "Modified_A2O"
    MULTISTAGE_A2O = "Multistage_A2O"
    AO = "AO"
    MODIFIED_AO = "Modified_AO"
    OXIDATION_DITCH = "Oxidation_Ditch"
    MODIFIED_OD = "Modified_Oxidation_Ditch"
    SBR = "SBR"
    MSBR = "MSBR"
    CASS = "CASS"
    CAST = "CAST"
    MBR = "MBR"
    UNITANK = "UNITANK"
    MBBR = "MBBR"
    BAF = "BAF"
    ACTIVATED_SLUDGE = "Activated_Sludge"
    CONTACT_OXIDATION = "Contact_Oxidation"
    OTHER = "Other"
    NONE = "None"

class AdvancedTreatment(Enum):
    """深度处理工艺标准枚举"""
    HIGH_EFFICIENCY_SEDIMENTATION = "High_Efficiency_Sedimentation"
    V_FILTER = "V_Filter"
    CLOTH_FILTER = "Cloth_Filter"
    DISK_FILTER = "Disk_Filter"
    SAND_FILTER = "Sand_Filter"
    DENITRIFICATION_FILTER = "Denitrification_Filter"
    DEEP_BED_FILTER = "Deep_Bed_Filter"
    MEMBRANE_FILTER = "Membrane_Filter"
    MBR_MEMBRANE = "MBR_Membrane"
    BAF = "BAF"
    OZONE = "Ozone"
    COMBINED_PROCESS = "Combined_Process"
    OTHER = "Other"
    NONE = "None"

class DisinfectionMethod(Enum):
    """消毒方式标准枚举"""
    SODIUM_HYPOCHLORITE = "Sodium_Hypochlorite"
    UV = "UV"
    UV_PLUS_SODIUM_HYPOCHLORITE = "UV_Plus_Sodium_Hypochlorite"
    CHLORINE = "Chlorine"
    CHLORINE_DIOXIDE = "Chlorine_Dioxide"
    OZONE = "Ozone"
    OTHER = "Other"
    NONE = "None"

def analyze_and_standardize():
    df = pd.read_csv('data/wwtp_summary_clean.csv')
    
    print("=" * 80)
    print("污水处理工艺三阶段标准化分析")
    print("=" * 80)
    
    # 1. 主处理工艺分析与标准化
    print("\n1. 主处理工艺 (treatment_process)")
    print("-" * 60)
    process_counts = df['treatment_process'].value_counts()
    print(f"原始类型数: {len(process_counts)}")
    print("\n前10种分布:")
    for process, count in process_counts.head(10).items():
        print(f"  {process}: {count}家")
    
    treatment_mapping = create_treatment_mapping()
    
    # 2. 深度处理工艺分析与标准化
    print("\n2. 深度处理工艺 (advanced_treatment_process)")
    print("-" * 60)
    advanced_counts = df['advanced_treatment_process'].value_counts()
    print(f"原始类型数: {len(advanced_counts)}")
    print("\n前10种分布:")
    for process, count in advanced_counts.head(10).items():
        print(f"  {process}: {count}家")
    
    advanced_mapping = create_advanced_mapping()
    
    # 3. 消毒方式分析与标准化
    print("\n3. 消毒方式 (disinfection_method)")
    print("-" * 60)
    disinfection_counts = df['disinfection_method'].value_counts()
    print(f"原始类型数: {len(disinfection_counts)}")
    print("\n前10种分布:")
    for method, count in disinfection_counts.head(10).items():
        print(f"  {method}: {count}家")
    
    disinfection_mapping = create_disinfection_mapping()
    
    # 应用标准化映射
    df['treatment_process_std'] = df['treatment_process'].map(
        lambda x: treatment_mapping.get(str(x).strip(), TreatmentProcess.OTHER.value) 
        if pd.notna(x) else TreatmentProcess.NONE.value
    )
    
    df['advanced_treatment_std'] = df['advanced_treatment_process'].map(
        lambda x: advanced_mapping.get(str(x).strip(), AdvancedTreatment.OTHER.value) 
        if pd.notna(x) else AdvancedTreatment.NONE.value
    )
    
    df['disinfection_method_std'] = df['disinfection_method'].map(
        lambda x: disinfection_mapping.get(str(x).strip(), DisinfectionMethod.OTHER.value) 
        if pd.notna(x) else DisinfectionMethod.NONE.value
    )
    
    # 输出标准化结果统计
    print("\n" + "=" * 80)
    print("标准化后的分布统计")
    print("=" * 80)
    
    print("\n主处理工艺标准化后分布:")
    treatment_std_counts = df['treatment_process_std'].value_counts()
    for process, count in treatment_std_counts.items():
        print(f"  {process}: {count}家 ({count/len(df)*100:.1f}%)")
    
    print("\n深度处理工艺标准化后分布:")
    advanced_std_counts = df['advanced_treatment_std'].value_counts()
    for process, count in advanced_std_counts.items():
        print(f"  {process}: {count}家 ({count/len(df)*100:.1f}%)")
    
    print("\n消毒方式标准化后分布:")
    disinfection_std_counts = df['disinfection_method_std'].value_counts()
    for method, count in disinfection_std_counts.items():
        print(f"  {method}: {count}家 ({count/len(df)*100:.1f}%)")
    
    # 保存结果
    output_file = 'data/wwtp_data_standardized.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n标准化数据已保存至: {output_file}")
    
    # 输出映射统计
    print("\n" + "=" * 80)
    print("标准化映射统计")
    print("=" * 80)
    print(f"主处理工艺: {len(process_counts)} -> {len(treatment_std_counts)} 种")
    print(f"深度处理工艺: {len(advanced_counts)} -> {len(advanced_std_counts)} 种")
    print(f"消毒方式: {len(disinfection_counts)} -> {len(disinfection_std_counts)} 种")
    
    return df

def create_treatment_mapping() -> Dict[str, str]:
    """创建主处理工艺映射表"""
    mapping = {
        # A2O系列
        'A2O': TreatmentProcess.A2O.value,
        'AAO': TreatmentProcess.A2O.value,
        'AAO工艺': TreatmentProcess.A2O.value,
        'AAO活性污泥法': TreatmentProcess.A2O.value,
        '倒置A2O': TreatmentProcess.A2O.value,
        '倒置AAO': TreatmentProcess.A2O.value,
        'AAO+二沉池': TreatmentProcess.A2O.value,
        'AAO+高密度沉淀': TreatmentProcess.A2O.value,
        'AAO+MBR': TreatmentProcess.A2O.value,
        'AAO+MBBR': TreatmentProcess.A2O.value,
        'AAOAO+MBR': TreatmentProcess.A2O.value,
        'A2O-SBR': TreatmentProcess.A2O.value,
        'A2O-SBR池': TreatmentProcess.A2O.value,
        'SBR+AAO': TreatmentProcess.A2O.value,
        
        # 改良型A2O
        '改良型A2/O': TreatmentProcess.MODIFIED_A2O.value,
        '强化脱氮改良型A2/O': TreatmentProcess.MODIFIED_A2O.value,
        '多段强化脱氮改良剂A2/O': TreatmentProcess.MODIFIED_A2O.value,
        '改良型AAO工艺': TreatmentProcess.MODIFIED_A2O.value,
        '改良AAO生化池': TreatmentProcess.MODIFIED_A2O.value,
        '改良型CAST工艺+AAO': TreatmentProcess.MODIFIED_A2O.value,
        'AAO+改良交替式+AAOAO活性污泥法': TreatmentProcess.MODIFIED_A2O.value,
        '改良交替式+AAOAO活性污泥法': TreatmentProcess.MODIFIED_A2O.value,
        '改良A2O+IFAS+SBR': TreatmentProcess.MODIFIED_A2O.value,
        '水解酸化+氧化沟+活性砂滤、改良型AAO+高效沉淀池+转盘过滤器、改良型AAO+深床滤池': TreatmentProcess.MODIFIED_A2O.value,
        
        # 多级A2O
        '三段式A2O': TreatmentProcess.MULTISTAGE_A2O.value,
        '多模式A2O': TreatmentProcess.MULTISTAGE_A2O.value,
        '分点进水倒置A2O': TreatmentProcess.MULTISTAGE_A2O.value,
        '多级AO+AAO': TreatmentProcess.MULTISTAGE_A2O.value,
        '强化脱氮改良型A2/O+多级AO生化池': TreatmentProcess.MULTISTAGE_A2O.value,
        '氧化沟+多膜A2O': TreatmentProcess.MULTISTAGE_A2O.value,
        
        # AO系列
        'AO': TreatmentProcess.AO.value,
        '三段式AO': TreatmentProcess.AO.value,
        '水解酸化+AO': TreatmentProcess.AO.value,
        '水解池+多级AO': TreatmentProcess.MODIFIED_AO.value,
        '水解池+AO-SBR': TreatmentProcess.MODIFIED_AO.value,
        'AO工艺改造+深床反硝化滤池': TreatmentProcess.MODIFIED_AO.value,
        'AO延时曝气+高效浅层气浮池+纤维转盘滤池+臭氧接触池': TreatmentProcess.MODIFIED_AO.value,
        
        # 氧化沟系列
        '氧化沟': TreatmentProcess.OXIDATION_DITCH.value,
        '三沟式氧化沟': TreatmentProcess.OXIDATION_DITCH.value,
        '卡鲁赛尔氧化沟': TreatmentProcess.OXIDATION_DITCH.value,
        '卡鲁赛尔氧化沟+微曝氧化沟': TreatmentProcess.OXIDATION_DITCH.value,
        '氧化沟+二沉池': TreatmentProcess.OXIDATION_DITCH.value,
        '氧化沟及其衍生工艺、MBR及其衍生工艺': TreatmentProcess.OXIDATION_DITCH.value,
        
        # 改良型氧化沟
        '改良型氧化沟': TreatmentProcess.MODIFIED_OD.value,
        '改良式氧化沟': TreatmentProcess.MODIFIED_OD.value,
        '水解酸化+改良型氧化沟': TreatmentProcess.MODIFIED_OD.value,
        
        # SBR系列
        'SBR及其衍生工艺': TreatmentProcess.SBR.value,
        '改良型SBR': TreatmentProcess.SBR.value,
        '初沉+NSBR': TreatmentProcess.SBR.value,
        'MSBR': TreatmentProcess.MSBR.value,
        
        # CAST/CASS
        'CAST': TreatmentProcess.CAST.value,
        'cast工艺': TreatmentProcess.CAST.value,
        
        # MBR
        'MBR及其衍生工艺': TreatmentProcess.MBR.value,
        
        # 其他工艺
        'UNITANK': TreatmentProcess.UNITANK.value,
        'MBBR': TreatmentProcess.MBBR.value,
        'Bardenphop+MBBR': TreatmentProcess.MBBR.value,
        
        # 特殊处理
        '预处理+延时爆气+芬顿氧化加气浮组合': TreatmentProcess.OTHER.value,
        '其它': TreatmentProcess.OTHER.value,
        '无': TreatmentProcess.NONE.value,
    }
    return mapping

def create_advanced_mapping() -> Dict[str, str]:
    """创建深度处理工艺映射表"""
    mapping = {
        # 高效沉淀池组合
        '高效沉淀+V型滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀池+V型滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀池+转盘滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀池+反硝化深床滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀池加反硝化深床滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀+深床滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高效沉淀池+滤布滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        '高密度沉淀+纤维转盘滤池': AdvancedTreatment.COMBINED_PROCESS.value,
        
        # 单一高效沉淀
        '高效沉淀池': AdvancedTreatment.HIGH_EFFICIENCY_SEDIMENTATION.value,
        '高效沉淀': AdvancedTreatment.HIGH_EFFICIENCY_SEDIMENTATION.value,
        '高密度沉淀池': AdvancedTreatment.HIGH_EFFICIENCY_SEDIMENTATION.value,
        
        # 滤池类
        'V型滤池': AdvancedTreatment.V_FILTER.value,
        '滤布滤池': AdvancedTreatment.CLOTH_FILTER.value,
        '纤维转盘滤池': AdvancedTreatment.DISK_FILTER.value,
        '转盘滤池': AdvancedTreatment.DISK_FILTER.value,
        '纤维转盘滤器': AdvancedTreatment.DISK_FILTER.value,
        '连续流砂滤池': AdvancedTreatment.SAND_FILTER.value,
        '活性砂滤池': AdvancedTreatment.SAND_FILTER.value,
        '活性砂过滤': AdvancedTreatment.SAND_FILTER.value,
        
        # 反硝化/深床滤池
        '反硝化深床滤池': AdvancedTreatment.DENITRIFICATION_FILTER.value,
        '深床滤池': AdvancedTreatment.DEEP_BED_FILTER.value,
        '深床反硝化滤池': AdvancedTreatment.DENITRIFICATION_FILTER.value,
        
        # 膜处理
        '膜过滤': AdvancedTreatment.MEMBRANE_FILTER.value,
        'MBR膜池': AdvancedTreatment.MBR_MEMBRANE.value,
        '曝气生物滤池+活性砂滤池、MBR膜池': AdvancedTreatment.MBR_MEMBRANE.value,
        
        # 曝气生物滤池
        '曝气生物滤池': AdvancedTreatment.BAF.value,
        '曝气生物滤池+活性砂滤池': AdvancedTreatment.BAF.value,
        
        # 臭氧
        '臭氧接触氧化': AdvancedTreatment.OZONE.value,
        '臭氧': AdvancedTreatment.OZONE.value,
        
        # 无或其他
        '无': AdvancedTreatment.NONE.value,
        '/': AdvancedTreatment.NONE.value,
    }
    return mapping

def create_disinfection_mapping() -> Dict[str, str]:
    """创建消毒方式映射表"""
    mapping = {
        # 次氯酸钠
        '次氯酸钠': DisinfectionMethod.SODIUM_HYPOCHLORITE.value,
        '次氯酸钠消毒': DisinfectionMethod.SODIUM_HYPOCHLORITE.value,
        '次氯酸钠药剂': DisinfectionMethod.SODIUM_HYPOCHLORITE.value,
        '二氧化氯和次氯酸钠': DisinfectionMethod.SODIUM_HYPOCHLORITE.value,
        
        # 紫外线
        '紫外线消毒': DisinfectionMethod.UV.value,
        '紫外消毒': DisinfectionMethod.UV.value,
        '紫外线': DisinfectionMethod.UV.value,
        '紫外': DisinfectionMethod.UV.value,
        
        # 紫外线+次氯酸钠
        '紫外线消毒+次氯酸钠': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '紫外线+次氯酸钠': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '紫外线消毒+次氯酸钠消毒': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '次氯酸钠、紫外线消毒': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '紫外线加次氯酸钠消毒': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '紫外+次氯酸钠': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '次氯酸钠+紫外': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        '次氯酸钠+紫外线': DisinfectionMethod.UV_PLUS_SODIUM_HYPOCHLORITE.value,
        
        # 液氯
        '液氯消毒': DisinfectionMethod.CHLORINE.value,
        '液氯': DisinfectionMethod.CHLORINE.value,
        
        # 二氧化氯
        '二氧化氯': DisinfectionMethod.CHLORINE_DIOXIDE.value,
        '二氧化氯消毒': DisinfectionMethod.CHLORINE_DIOXIDE.value,
        
        # 臭氧
        '臭氧消毒': DisinfectionMethod.OZONE.value,
        '臭氧': DisinfectionMethod.OZONE.value,
        
        # 无或其他
        '无': DisinfectionMethod.NONE.value,
        '/': DisinfectionMethod.NONE.value,
    }
    return mapping

if __name__ == "__main__":
    df = analyze_and_standardize()