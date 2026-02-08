"""
工具函数
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_data(file_path: str | Path) -> pd.DataFrame:
    """加载数据文件"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path, sep=' ')
    elif file_path.suffix == '.zip':
        return pd.read_csv(file_path, compression='zip', sep=' ')
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """减少 DataFrame 内存占用"""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    
    if verbose:
        print(f'内存优化: {start_mem:.2f} MB -> {end_mem:.2f} MB '
              f'(减少 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df


def save_submission(predictions: np.ndarray, test_df: pd.DataFrame, 
                    filename: str, submissions_dir: str | Path) -> Path:
    """保存提交文件"""
    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    submission = pd.DataFrame({
        'SaleID': test_df['SaleID'],
        'price': predictions
    })
    
    # 确保 price 非负
    submission['price'] = submission['price'].clip(lower=0)
    
    filepath = submissions_dir / filename
    submission.to_csv(filepath, index=False)
    print(f"提交文件已保存: {filepath}")
    
    return filepath


def print_feature_importance(model, feature_names: list, top_n: int = 20):
    """打印特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importance = model.feature_importance()
    else:
        print("模型不支持特征重要性")
        return
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} 特征重要性:")
    print(feat_imp.head(top_n).to_string(index=False))
    
    return feat_imp
