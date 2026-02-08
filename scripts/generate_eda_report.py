"""
生成 EDA 报告
"""
import pandas as pd
import numpy as np
import os

# 创建报告目录
os.makedirs("reports", exist_ok=True)

# 加载数据
train = pd.read_csv("data/raw/used_car_train_20200313.csv", sep=" ")
test = pd.read_csv("data/raw/used_car_testA_20200313.csv", sep=" ")

# 生成报告
report = []
report.append("# EDA 探索性数据分析报告\n")

report.append("## 1. 数据概览\n")
report.append(f"| 数据集 | 行数 | 列数 |")
report.append(f"|--------|------|------|")
report.append(f"| 训练集 | {train.shape[0]:,} | {train.shape[1]} |")
report.append(f"| 测试集 | {test.shape[0]:,} | {test.shape[1]} |")
report.append("")

report.append("### 列名")
report.append(f"```")
report.append(", ".join(train.columns.tolist()))
report.append(f"```\n")

report.append("## 2. 目标变量 price 分析\n")
report.append(f"| 统计量 | 值 |")
report.append(f"|--------|-----|")
report.append(f"| 均值 | {train['price'].mean():,.2f} |")
report.append(f"| 中位数 | {train['price'].median():,.2f} |")
report.append(f"| 标准差 | {train['price'].std():,.2f} |")
report.append(f"| 最小值 | {train['price'].min():,} |")
report.append(f"| 最大值 | {train['price'].max():,} |")
report.append(f"| 偏度 | {train['price'].skew():.2f} |")
report.append("")
report.append("> ⚠️ **发现**: price 偏度较高，建议使用 `log1p()` 变换\n")

report.append("## 3. 分类特征分析\n")
report.append(f"| 特征 | 唯一值数量 |")
report.append(f"|------|------------|")
cat_cols = ["brand", "model", "bodyType", "fuelType", "gearbox", "regionCode", "seller", "offerType"]
for col in cat_cols:
    report.append(f"| {col} | {train[col].nunique()} |")
report.append("")

report.append("## 4. 缺失值分析\n")
report.append(f"| 特征 | 缺失数量 | 缺失比例 |")
report.append(f"|------|----------|----------|")
report.append(f"| bodyType | {train['bodyType'].isnull().sum()} | {train['bodyType'].isnull().mean()*100:.2f}% |")
report.append(f"| fuelType | {train['fuelType'].isnull().sum()} | {train['fuelType'].isnull().mean()*100:.2f}% |")
report.append(f"| gearbox | {train['gearbox'].isnull().sum()} | {train['gearbox'].isnull().mean()*100:.2f}% |")
nrd_special = (train["notRepairedDamage"] == "-").sum()
report.append(f"| notRepairedDamage (特殊值'-') | {nrd_special} | {nrd_special/len(train)*100:.2f}% |")
report.append("")

report.append("## 5. power 异常值分析\n")
report.append(f"| 统计量 | 值 |")
report.append(f"|--------|-----|")
report.append(f"| 均值 | {train['power'].mean():.2f} |")
report.append(f"| 中位数 | {train['power'].median():.2f} |")
report.append(f"| 最大值 | {train['power'].max()} |")
report.append(f"| power > 600 的数量 | {(train['power'] > 600).sum()} |")
report.append("")
report.append("> ⚠️ **发现**: 存在 power 异常大的值，建议截断到 600\n")

report.append("## 6. 匿名特征与 price 相关性\n")
v_cols = [f"v_{i}" for i in range(15)]
corr = train[v_cols + ["price"]].corr()["price"].drop("price").abs().sort_values(ascending=False)
report.append(f"| 特征 | 相关性 (绝对值) |")
report.append(f"|------|------------------|")
for feat, val in corr.head(10).items():
    report.append(f"| {feat} | {val:.4f} |")
report.append("")

report.append("## 7. EDA 结论\n")
report.append("### 需要处理的问题:")
report.append("1. **price 偏态分布** - 使用 `log1p()` 变换")
report.append("2. **power 异常值** - 截断到 600")
report.append("3. **notRepairedDamage 特殊值** - 将 '-' 替换为 NaN")
report.append("4. **缺失值填充** - bodyType, fuelType, gearbox 用众数填充")
report.append("")
report.append("### 重要特征:")
report.append("- 匿名特征 v_0, v_8, v_12 与 price 相关性较高")
report.append("- 车龄 (需要从日期计算) 是重要特征")
report.append("- power, kilometer 与价格相关")

# 保存报告
with open("reports/eda_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("EDA 报告已生成: reports/eda_report.md")
print("\n" + "="*50)
print("关键发现摘要:")
print("="*50)
print(f"训练集: {train.shape[0]:,} 行")
print(f"price 偏度: {train['price'].skew():.2f}")
print(f"power 异常值 (>600): {(train['power'] > 600).sum()}")
print(f"notRepairedDamage 特殊值: {nrd_special}")
