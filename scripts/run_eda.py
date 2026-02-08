"""
EDA 探索性数据分析脚本
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("加载数据...")
print("=" * 60)

train = pd.read_csv("data/raw/used_car_train_20200313.csv", sep=" ")
test = pd.read_csv("data/raw/used_car_testA_20200313.csv", sep=" ")

print(f"训练集: {train.shape[0]} 行, {train.shape[1]} 列")
print(f"测试集: {test.shape[0]} 行, {test.shape[1]} 列")

print("\n" + "=" * 60)
print("列名:")
print("=" * 60)
print(list(train.columns))

print("\n" + "=" * 60)
print("目标变量 price 统计:")
print("=" * 60)
print(train["price"].describe())
print(f"\n偏度 (skewness): {train['price'].skew():.2f}")
print("(偏度>0 说明右偏，建议使用 log1p 变换)")

print("\n" + "=" * 60)
print("分类特征基数统计:")
print("=" * 60)
cat_cols = ["brand", "model", "bodyType", "fuelType", "gearbox", "regionCode"]
for col in cat_cols:
    print(f"  {col}: {train[col].nunique()} 个唯一值")

print("\n" + "=" * 60)
print("缺失值统计:")
print("=" * 60)
print(f"  bodyType 缺失: {train['bodyType'].isnull().sum()}")
print(f"  fuelType 缺失: {train['fuelType'].isnull().sum()}")
print(f"  gearbox 缺失: {train['gearbox'].isnull().sum()}")
nrd_special = (train["notRepairedDamage"] == "-").sum()
print(f"  notRepairedDamage 特殊值'-': {nrd_special}")

print("\n" + "=" * 60)
print("power 异常值分析:")
print("=" * 60)
print(f"  power > 600 的数量: {(train['power'] > 600).sum()}")
print(f"  power 最大值: {train['power'].max()}")

print("\n" + "=" * 60)
print("匿名特征 v_0 ~ v_14 与 price 相关性 (Top 5):")
print("=" * 60)
v_cols = [f"v_{i}" for i in range(15)]
corr_with_price = train[v_cols + ["price"]].corr()["price"].drop("price").abs().sort_values(ascending=False)
print(corr_with_price.head())

print("\n" + "=" * 60)
print("EDA 完成!")
print("=" * 60)
