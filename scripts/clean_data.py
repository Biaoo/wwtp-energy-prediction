import pandas as pd
import numpy as np

def clean_wwtp_data():
    df = pd.read_excel('data/2022-wwt-data.xlsx', sheet_name='汇总')
    
    df = df.iloc[2:].reset_index(drop=True)
    
    column_mapping = {
        '序号': 'id',
        '所在市': 'city',
        '所在区（县）': 'district',
        '排水单位名称': 'drainage_company',
        '污水处理厂名称': 'wwtp_name',
        '处理能力（万立方米/日）': 'treatment_capacity_10k_m3_per_day',
        '处理水量（万立方米/年）': 'annual_treatment_volume_10k_m3',
        '污水处理工艺': 'treatment_process',
        'Unnamed: 8': 'advanced_treatment_process',
        'Unnamed: 9': 'disinfection_method',
        '污水处理-水质指标（mg/L）': 'cod_influent_mg_l',
        'Unnamed: 11': 'cod_effluent_mg_l',
        'Unnamed: 12': 'bod5_influent_mg_l',
        'Unnamed: 13': 'bod5_effluent_mg_l',
        'Unnamed: 14': 'ss_influent_mg_l',
        'Unnamed: 15': 'ss_effluent_mg_l',
        '污水处理-水质指标（mg/L）.1': 'nh3n_influent_mg_l',
        'Unnamed: 17': 'nh3n_effluent_mg_l',
        'Unnamed: 18': 'tn_influent_mg_l',
        'Unnamed: 19': 'tn_effluent_mg_l',
        'Unnamed: 20': 'tp_influent_mg_l',
        'Unnamed: 21': 'tp_effluent_mg_l',
        '累计用电率（kWh）': 'annual_electricity_consumption_kwh',
        '污水处理工艺.1': 'sludge_treatment_process'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    numeric_columns = [
        'id', 
        'treatment_capacity_10k_m3_per_day',
        'annual_treatment_volume_10k_m3',
        'cod_influent_mg_l', 'cod_effluent_mg_l',
        'bod5_influent_mg_l', 'bod5_effluent_mg_l',
        'ss_influent_mg_l', 'ss_effluent_mg_l',
        'nh3n_influent_mg_l', 'nh3n_effluent_mg_l',
        'tn_influent_mg_l', 'tn_effluent_mg_l',
        'tp_influent_mg_l', 'tp_effluent_mg_l',
        'annual_electricity_consumption_kwh'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['id', 'wwtp_name'])
    
    df['id'] = df['id'].astype('Int64')
    
    output_file = 'data/wwtp_summary_clean.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"数据清洗完成！")
    print(f"原始数据行数: 95")
    print(f"清洗后数据行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"\n已保存至: {output_file}")
    
    print("\n数据预览:")
    print(df.head())
    
    print("\n列信息:")
    print(df.dtypes)
    
    print("\n数据统计摘要:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    clean_wwtp_data()