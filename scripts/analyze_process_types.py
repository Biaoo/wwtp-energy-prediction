import pandas as pd
import re
from collections import Counter

def analyze_treatment_processes():
    df = pd.read_csv('data/wwtp_summary_clean.csv')
    
    print("=" * 80)
    print("污水处理工艺类型分析")
    print("=" * 80)
    
    print("\n1. 主处理工艺分布")
    print("-" * 60)
    process_counts = df['treatment_process'].value_counts()
    print(f"总共有 {len(process_counts)} 种不同的工艺描述\n")
    
    for process, count in process_counts.items():
        if pd.notna(process):
            print(f"  {process}: {count} 家 ({count/len(df)*100:.1f}%)")
    
    print("\n2. 工艺名称相似性分析")
    print("-" * 60)
    
    process_list = df['treatment_process'].dropna().unique()
    
    similar_groups = {
        'A2O系列': [],
        'A/O系列': [],
        '氧化沟系列': [],
        'SBR系列': [],
        'MBR系列': [],
        'CAST系列': [],
        'UNITANK系列': [],
        '生物滤池系列': [],
        '接触氧化系列': [],
        '其他': []
    }
    
    for process in process_list:
        process_upper = str(process).upper()
        
        if 'A2O' in process_upper or 'A2/O' in process_upper or 'A²O' in process_upper or 'AAO' in process_upper:
            similar_groups['A2O系列'].append(process)
        elif 'A/O' in process_upper and 'A2' not in process_upper:
            similar_groups['A/O系列'].append(process)
        elif '氧化沟' in process or 'OXIDATION DITCH' in process_upper:
            similar_groups['氧化沟系列'].append(process)
        elif 'SBR' in process_upper:
            similar_groups['SBR系列'].append(process)
        elif 'MBR' in process_upper:
            similar_groups['MBR系列'].append(process)
        elif 'CAST' in process_upper:
            similar_groups['CAST系列'].append(process)
        elif 'UNITANK' in process_upper:
            similar_groups['UNITANK系列'].append(process)
        elif '滤池' in process or 'FILTER' in process_upper:
            similar_groups['生物滤池系列'].append(process)
        elif '接触氧化' in process:
            similar_groups['接触氧化系列'].append(process)
        else:
            similar_groups['其他'].append(process)
    
    print("\n工艺分组结果：")
    for group, processes in similar_groups.items():
        if processes:
            print(f"\n{group} ({len(processes)}种):")
            for p in processes:
                count = process_counts[p]
                print(f"  • {p} ({count}家)")
    
    print("\n3. 建议的标准化映射")
    print("-" * 60)
    
    standardization_map = {}
    
    for process in process_list:
        process_upper = str(process).upper()
        
        if 'A2O' in process_upper or 'A2/O' in process_upper or 'A²O' in process_upper or 'AAO' in process_upper:
            if '多级' in process:
                standardization_map[process] = 'Multi-stage A2O'
            elif '改良' in process or '强化' in process:
                standardization_map[process] = 'Modified A2O'
            else:
                standardization_map[process] = 'A2O'
        elif 'A/O' in process_upper and 'A2' not in process_upper:
            if '多级' in process:
                standardization_map[process] = 'Multi-stage A/O'
            else:
                standardization_map[process] = 'A/O'
        elif '氧化沟' in process:
            if 'ORBAL' in process_upper:
                standardization_map[process] = 'Orbal Oxidation Ditch'
            elif 'CARROUSEL' in process_upper or '卡鲁塞尔' in process:
                standardization_map[process] = 'Carrousel Oxidation Ditch'
            elif 'DE' in process_upper:
                standardization_map[process] = 'DE Oxidation Ditch'
            elif '改良' in process:
                standardization_map[process] = 'Modified Oxidation Ditch'
            else:
                standardization_map[process] = 'Oxidation Ditch'
        elif 'SBR' in process_upper:
            if 'ICEAS' in process_upper:
                standardization_map[process] = 'ICEAS'
            elif 'CASS' in process_upper:
                standardization_map[process] = 'CASS'
            elif 'MSBR' in process_upper:
                standardization_map[process] = 'MSBR'
            else:
                standardization_map[process] = 'SBR'
        elif 'MBR' in process_upper:
            standardization_map[process] = 'MBR'
        elif 'CAST' in process_upper:
            standardization_map[process] = 'CAST'
        elif 'UNITANK' in process_upper:
            standardization_map[process] = 'UNITANK'
        elif '曝气生物滤池' in process or 'BAF' in process_upper:
            standardization_map[process] = 'BAF'
        elif '生物滤池' in process:
            standardization_map[process] = 'Biofilter'
        elif '接触氧化' in process:
            standardization_map[process] = 'Contact Oxidation'
        elif '活性污泥' in process:
            standardization_map[process] = 'Activated Sludge'
        else:
            standardization_map[process] = process
    
    print("\n标准化映射表（原始 -> 标准化）：")
    for original, standardized in sorted(standardization_map.items(), key=lambda x: x[1]):
        if original != standardized:
            count = process_counts[original]
            print(f"  {original} -> {standardized} ({count}家)")
    
    df['treatment_process_standardized'] = df['treatment_process'].map(
        lambda x: standardization_map.get(x, x) if pd.notna(x) else x
    )
    
    print("\n4. 标准化后的工艺分布")
    print("-" * 60)
    standardized_counts = df['treatment_process_standardized'].value_counts()
    print(f"标准化后共有 {len(standardized_counts)} 种工艺类型\n")
    
    for process, count in standardized_counts.head(15).items():
        if pd.notna(process):
            print(f"  {process}: {count} 家 ({count/len(df)*100:.1f}%)")
    
    print("\n5. 高级处理工艺分析")
    print("-" * 60)
    advanced_counts = df['advanced_treatment_process'].value_counts()
    print(f"共有 {len(advanced_counts)} 种高级处理工艺\n")
    
    for process, count in advanced_counts.head(10).items():
        if pd.notna(process):
            print(f"  {process}: {count} 家")
    
    print("\n6. 消毒方式分析")
    print("-" * 60)
    disinfection_counts = df['disinfection_method'].value_counts()
    print(f"共有 {len(disinfection_counts)} 种消毒方式\n")
    
    for method, count in disinfection_counts.head(10).items():
        if pd.notna(method):
            print(f"  {method}: {count} 家")
    
    df.to_csv('data/wwtp_summary_standardized.csv', index=False, encoding='utf-8-sig')
    print(f"\n标准化后的数据已保存至: data/wwtp_summary_standardized.csv")
    
    return df, standardization_map

if __name__ == "__main__":
    df, mapping = analyze_treatment_processes()