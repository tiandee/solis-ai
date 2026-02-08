"""
特征工程模块
"""
import pandas as pd
import numpy as np
from typing import Tuple


def parse_date_features(df: pd.DataFrame, date_cols: list) -> pd.DataFrame:
    """解析日期特征，提取年、月、日"""
    df = df.copy()
    
    for col in date_cols:
        if col not in df.columns:
            continue
            
        # 转换为字符串并处理
        df[col] = df[col].astype(str)
        
        # 提取年月日
        df[f'{col}_year'] = df[col].str[:4].astype(int)
        df[f'{col}_month'] = df[col].str[4:6].astype(int)
        df[f'{col}_day'] = df[col].str[6:8].astype(int)
    
    return df


def create_car_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算车龄相关特征"""
    df = df.copy()
    
    # 确保日期特征已解析
    if 'regDate_year' not in df.columns:
        df = parse_date_features(df, ['regDate', 'creatDate'])
    
    # 车龄（年）
    df['car_age_year'] = df['creatDate_year'] - df['regDate_year']
    
    # 车龄（月）
    df['car_age_month'] = (
        (df['creatDate_year'] - df['regDate_year']) * 12 +
        (df['creatDate_month'] - df['regDate_month'])
    )
    
    # 处理异常值
    df['car_age_year'] = df['car_age_year'].clip(lower=0)
    df['car_age_month'] = df['car_age_month'].clip(lower=0)
    
    return df


def create_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """处理 power 特征"""
    df = df.copy()
    
    # 处理异常值：power 大于 600 的截断
    df['power'] = df['power'].clip(upper=600)
    
    # power 分箱
    df['power_bin'] = pd.cut(df['power'], bins=[0, 50, 100, 150, 200, 300, 600], 
                             labels=[0, 1, 2, 3, 4, 5])
    df['power_bin'] = df['power_bin'].astype(float).fillna(-1).astype(int)
    
    return df


def create_kilometer_features(df: pd.DataFrame) -> pd.DataFrame:
    """处理 kilometer 特征"""
    df = df.copy()
    
    # kilometer 分箱
    df['kilometer_bin'] = pd.cut(df['kilometer'], bins=[0, 3, 6, 9, 12, 15, 20], 
                                  labels=[0, 1, 2, 3, 4, 5])
    df['kilometer_bin'] = df['kilometer_bin'].astype(float).fillna(-1).astype(int)
    
    # 每年行驶公里数
    df['km_per_year'] = df['kilometer'] / (df['car_age_year'] + 1)
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值"""
    df = df.copy()
    
    # 处理 notRepairedDamage 中的 '-'
    if 'notRepairedDamage' in df.columns:
        df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
        df['notRepairedDamage'] = df['notRepairedDamage'].astype(float)
    
    # 数值特征用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # 分类特征用众数填充
    cat_cols = ['bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else -1)
    
    return df


def create_statistical_features(df: pd.DataFrame, group_cols: list, 
                                 target_col: str = None) -> pd.DataFrame:
    """创建统计特征"""
    df = df.copy()
    
    # 基于 brand 的统计
    if 'brand' in df.columns:
        brand_count = df.groupby('brand').size().reset_index(name='brand_count')
        df = df.merge(brand_count, on='brand', how='left')
    
    # 基于 model 的统计
    if 'model' in df.columns:
        model_count = df.groupby('model').size().reset_index(name='model_count')
        df = df.merge(model_count, on='model', how='left')
    
    # 基于 regionCode 的统计
    if 'regionCode' in df.columns:
        region_count = df.groupby('regionCode').size().reset_index(name='region_count')
        df = df.merge(region_count, on='regionCode', how='left')
    
    return df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    完整的特征工程 Pipeline
    
    Args:
        train_df: 训练数据
        test_df: 测试数据（可选）
    
    Returns:
        处理后的训练集和测试集
    """
    # 合并处理
    if test_df is not None:
        test_df['price'] = -1  # 占位
        df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        train_size = len(train_df)
    else:
        df = train_df.copy()
        train_size = len(df)
    
    # 1. 解析日期
    df = parse_date_features(df, ['regDate', 'creatDate'])
    
    # 2. 车龄特征
    df = create_car_age_features(df)
    
    # 3. Power 特征
    df = create_power_features(df)
    
    # 4. Kilometer 特征
    df = create_kilometer_features(df)
    
    # 5. 处理缺失值
    df = handle_missing_values(df)
    
    # 6. 统计特征
    df = create_statistical_features(df, ['brand', 'model', 'regionCode'])
    
    # 分割回训练集和测试集
    train_processed = df.iloc[:train_size].copy()
    test_processed = df.iloc[train_size:].copy() if test_df is not None else None
    
    if test_processed is not None:
        test_processed = test_processed.drop('price', axis=1)
    
    return train_processed, test_processed
