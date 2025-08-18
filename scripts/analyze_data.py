import pandas as pd
import numpy as np
from pathlib import Path

def analyze_excel_file(file_path):
    xlsx_file = pd.ExcelFile(file_path)
    
    print("=" * 80)
    print(f"Excel文件分析报告: {Path(file_path).name}")
    print("=" * 80)
    print(f"\n文件包含 {len(xlsx_file.sheet_names)} 个工作表:")
    for i, sheet in enumerate(xlsx_file.sheet_names, 1):
        print(f"  {i}. {sheet}")
    
    print("\n" + "=" * 80)
    
    all_sheets_data = {}
    
    for sheet_name in xlsx_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        all_sheets_data[sheet_name] = df
        
        print(f"\n【工作表: {sheet_name}】")
        print("-" * 60)
        
        print(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        print(f"\n列信息:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  • {col}")
            print(f"    - 数据类型: {dtype}")
            print(f"    - 非空值数: {non_null}/{len(df)} (缺失率: {null_pct:.1f}%)")
            
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"    - 范围: [{df[col].min():.2f}, {df[col].max():.2f}]")
                print(f"    - 均值: {df[col].mean():.2f}, 标准差: {df[col].std():.2f}")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"    - 时间范围: {df[col].min()} 至 {df[col].max()}")
            else:
                unique_count = df[col].nunique()
                print(f"    - 唯一值数量: {unique_count}")
                if unique_count <= 10:
                    print(f"    - 唯一值: {df[col].unique().tolist()}")
        
        print(f"\n数据预览 (前5行):")
        print(df.head().to_string())
        
        if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) or 'date' in df.columns[0].lower() or '日期' in df.columns[0]:
            print(f"\n时间序列特征:")
            date_col = df.columns[0]
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            if df[date_col].notna().any():
                print(f"  • 时间跨度: {df[date_col].min()} 至 {df[date_col].max()}")
                time_diff = df[date_col].diff().dropna()
                if len(time_diff) > 0:
                    common_interval = time_diff.mode().iloc[0] if not time_diff.mode().empty else time_diff.median()
                    print(f"  • 数据频率: {common_interval}")
    
    print("\n" + "=" * 80)
    print("【数据关联性分析】")
    print("-" * 60)
    
    sheet_names = list(all_sheets_data.keys())
    for i in range(len(sheet_names)):
        for j in range(i+1, len(sheet_names)):
            sheet1, sheet2 = sheet_names[i], sheet_names[j]
            df1, df2 = all_sheets_data[sheet1], all_sheets_data[sheet2]
            
            common_cols = set(df1.columns) & set(df2.columns)
            if common_cols:
                print(f"\n{sheet1} 与 {sheet2} 的共同列: {list(common_cols)}")
    
    print("\n" + "=" * 80)
    print("【数据质量总结】")
    print("-" * 60)
    
    for sheet_name, df in all_sheets_data.items():
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        print(f"\n{sheet_name}:")
        print(f"  • 数据完整性: {completeness:.1f}%")
        print(f"  • 缺失值总数: {missing_cells}/{total_cells}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"  • 数值列数量: {len(numeric_cols)}/{df.shape[1]}")
    
    return all_sheets_data

if __name__ == "__main__":
    file_path = "data/2022-wwt-data.xlsx"
    data = analyze_excel_file(file_path)