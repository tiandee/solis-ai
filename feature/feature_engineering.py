"""
特征工程模块
按照比赛规范放置在 feature/ 目录
"""
import pandas as pd
import numpy as np


def preprocess_special_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理特殊值"""
    df = df.copy()
    # notRepairedDamage 的 '-' 替换为 NaN
    df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
    df["notRepairedDamage"] = df["notRepairedDamage"].astype(float)
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """处理异常值"""
    df = df.copy()
    # power 截断到 [0, 600]
    df["power"] = df["power"].clip(lower=0, upper=600)
    return df


def parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """解析日期特征"""
    df = df.copy()
    
    df["regDate"] = df["regDate"].astype(str)
    df["creatDate"] = df["creatDate"].astype(str)
    
    df["regDate_year"] = df["regDate"].str[:4].astype(int)
    df["regDate_month"] = df["regDate"].str[4:6].astype(int)
    df["creatDate_year"] = df["creatDate"].str[:4].astype(int)
    df["creatDate_month"] = df["creatDate"].str[4:6].astype(int)
    
    return df


def create_car_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建车龄特征"""
    df = df.copy()
    
    df["car_age_year"] = df["creatDate_year"] - df["regDate_year"]
    df["car_age_month"] = (df["creatDate_year"] - df["regDate_year"]) * 12 + \
                          (df["creatDate_month"] - df["regDate_month"])
    
    df["car_age_year"] = df["car_age_year"].clip(lower=0)
    df["car_age_month"] = df["car_age_month"].clip(lower=0)
    
    return df


def create_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建分箱特征"""
    df = df.copy()
    
    # Power 分箱
    df["power_bin"] = pd.cut(
        df["power"], 
        bins=[0, 50, 100, 150, 200, 300, 600],
        labels=[0, 1, 2, 3, 4, 5]
    )
    df["power_bin"] = df["power_bin"].astype(float).fillna(-1).astype(int)
    
    # Kilometer 分箱
    df["kilometer_bin"] = pd.cut(
        df["kilometer"], 
        bins=[0, 3, 6, 9, 12, 15, 20],
        labels=[0, 1, 2, 3, 4, 5]
    )
    df["kilometer_bin"] = df["kilometer_bin"].astype(float).fillna(-1).astype(int)
    
    return df


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建比率特征"""
    df = df.copy()
    df["km_per_year"] = df["kilometer"] / (df["car_age_year"] + 1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """完整的特征工程流水线"""
    df = preprocess_special_values(df)
    df = clip_outliers(df)
    df = parse_date_features(df)
    df = create_car_age_features(df)
    df = create_binning_features(df)
    df = create_ratio_features(df)
    return df


if __name__ == "__main__":
    # 测试特征工程
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ", nrows=1000)
    train = build_features(train)
    print(f"特征工程完成，共 {len(train.columns)} 列")
    print(train.head())
