import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import sys,os

# =============================================================================
# --- 用户配置区域 ---
# =============================================================================

# 1. 【必须修改】输入您的原始CSV文件路径
INPUT_CSV_PATH = 'dataset_new/cl_train/raw/cl_train.csv' 

# 2. 【建议修改】输出文件的路径
OUTPUT_CSV_PATH = 'dataset_new_desc/cl_train/raw/cl_train.csv' 

# 3. 【必须修改】您的文件中包含SMILES的列的准确名称
SMILES_COL = 'smiles'

# 4. (可选) 清洗阈值
#    如果一个描述符在超过5%的分子中计算失败(NaN, inf, 或超大值)，就删除这个描述符列
BAD_DESCRIPTOR_THRESHOLD = 0.05 
#    定义“超大值”的判断阈值 (通常不需要修改)
LARGE_VALUE_THRESHOLD = 1e6

# =============================================================================
# --- 核心功能函数 ---
# =============================================================================

def calculate_rdkit_descriptors(smiles_series):
    """为一个包含SMILES的Pandas Series计算RDKit 2D描述符。"""
    print("--> 步骤 2/5: 开始计算RDKit描述符...")
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptors_list = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            descriptors_list.append([np.nan] * len(descriptor_names))
        else:
            descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
            descriptors_list.append(descriptors)
    print(f"    描述符计算完成。")
    return pd.DataFrame(descriptors_list, columns=descriptor_names, index=smiles_series.index)


def run_preprocessing_pipeline():
    """执行完整的数据加载、计算、清洗和保存流程。"""
    
    # 1. 加载原始数据
    print(f"--> 步骤 1/5: 开始加载原始数据文件: '{INPUT_CSV_PATH}'")
    try:
        df_original = pd.read_csv(INPUT_CSV_PATH)
        original_shape = df_original.shape
        print(f"    成功加载数据。原始数据包含 {original_shape[0]} 行, {original_shape[1]} 列。")
    except FileNotFoundError:
        print(f"    [错误] 找不到文件 '{INPUT_CSV_PATH}'。")
        sys.exit()

    if SMILES_COL not in df_original.columns:
        print(f"    [错误] 在CSV文件中找不到指定的SMILES列 '{SMILES_COL}'。")
        sys.exit()

    # 2. 计算描述符
    df_descriptors = calculate_rdkit_descriptors(df_original[SMILES_COL])

    # 3. 合并数据
    print("--> 步骤 3/5: 合并原始数据和描述符...")
    df_processed = pd.concat([df_original, df_descriptors], axis=1)
    print("    数据合并完成。")
    df_processed=df_processed.drop(columns=['Ipc'])
    # 4. 执行包含“超大值”检测的最终清洗流程
    print("--> 步骤 4/5: 开始执行最终数据清洗流程...")
    
    # =============================================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼  【最终版清洗逻辑】  ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # =============================================================================
    # 4.1 统一问题：将inf和超大值全部转换为NaN
    
    # 筛选出数值类型的列进行操作
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    
    # 替换无穷大值
    df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print("    [清洗-0] 已将所有 inf 值替换为 NaN。")
    
    # 替换超大值
    # 创建一个布尔掩码，标记所有绝对值超过阈值的位置
    mask_large_values = (df_processed[numeric_cols].abs() > LARGE_VALUE_THRESHOLD)
    # 将这些位置的值设为NaN
    df_processed[mask_large_values] = np.nan
    print("    [清洗-0] 已将所有超大值替换为 NaN。")

    # 4.2 第一步：识别并删除“坏”的描述符（列）
    # 现在所有的坏数据点都是NaN了，逻辑变得简单
    missing_ratios = df_processed.isnull().sum() / len(df_processed)
    bad_cols = missing_ratios[missing_ratios > BAD_DESCRIPTOR_THRESHOLD].index.tolist()
    
    # 从待删除列表中移除原始数据中的列，以防万一
    original_cols_to_keep = df_original.columns.tolist()
    bad_cols = [col for col in bad_cols if col not in original_cols_to_keep]

    if bad_cols:
        print(f"    [清洗-1] 发现 {len(bad_cols)} 个描述符列因包含过多无效值(NaN/inf/超大值)将被删除:")
        print(f"      {bad_cols}")
        df_processed.drop(columns=bad_cols, inplace=True)
    else:
        print("    [清洗-1] 未发现需要删除的‘坏’描述符列。")

    # 4.3 第二步：删除剩余包含零星NaN的分子（行）
    initial_rows = len(df_processed)
    df_processed.dropna(inplace=True)
    final_rows = len(df_processed)
    
    print(f"    [清洗-2] 移除了 {initial_rows - final_rows} 个包含零星无效值的分子（行）。")
    print("    数据清洗完成。")
    # =============================================================================
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲  【清洗结束】  ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # =============================================================================

    # 5. 保存最终结果
    print(f"--> 步骤 5/5: 保存清理后的数据到 '{OUTPUT_CSV_PATH}'")
    try:
        dir = os.path.dirname(OUTPUT_CSV_PATH)
        os.makedirs(dir, exist_ok=True)
        df_processed.to_csv(OUTPUT_CSV_PATH, index=False)
        final_shape = df_processed.shape
        print("\n🎉 处理完成！")
        print(f"    原始数据尺寸: {original_shape[0]} 行 x {original_shape[1]} 列")
        print(f"    最终输出尺寸: {final_shape[0]} 行 x {final_shape[1]} 列")
        print(f"    结果已保存到: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"    [错误] 保存文件时发生错误: {e}")

# =============================================================================
# --- 脚本执行入口 ---
# =============================================================================
if __name__ == "__main__":
    run_preprocessing_pipeline()
