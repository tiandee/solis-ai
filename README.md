# 天池二手车价格预测 - 比赛解决方案

## 解决方案概述

本方案采用 **LightGBM + 5折交叉验证** 进行二手车价格预测。

### 算法流程

```
原始数据 → 特征工程 → LightGBM 5-Fold CV → 预测结果
```

## 特征工程

### 1. 数据清洗
- `notRepairedDamage` 特殊值 `-` 替换为 NaN
- `power` 异常值截断到 [0, 600]

### 2. 日期特征
- 从 `regDate` 提取注册年份、月份
- 从 `creatDate` 提取发布年份、月份
- 计算车龄（年、月）

### 3. 分箱特征
- `power_bin`: 功率分箱 (0-50, 50-100, ...)
- `kilometer_bin`: 里程分箱

### 4. 比率特征
- `km_per_year`: 年均行驶里程

## 模型参数

```python
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1
}
```

## 运行方式

```bash
cd code
python main.py
```

输出文件: `prediction_result/predictions.csv`

## 目录结构

```
project/
├── README.md              # 本文件
├── data/                  # 数据目录
│   ├── used_car_train_20200313.csv
│   └── used_car_testA_20200313.csv
├── feature/               # 特征工程
│   └── feature_engineering.py
├── model/                 # 模型
│   ├── train_model.py
│   └── lgb_fold*.pkl
├── code/                  # 主代码
│   ├── main.py
│   └── requirements.txt
├── prediction_result/     # 预测结果
│   └── predictions.csv
└── user_data/             # 中间数据
```

## 验证结果

- 5折交叉验证 MAE: ~670 (原始空间)
- 评估指标: Mean Absolute Error (MAE)
