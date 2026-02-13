# 二手车价格预测优化方案 (v2)

## 1. 成果摘要
- **优化前 (Baseline)**: MAE **497.39** (单模型 LightGBM)
- **优化后 (v2)**: MAE **445.68** (多模型融合 + 增强特征)
- **提升幅度**: **51.71 分** (显著提升)
- **当前排名**: ~2100 (天池长期赛)

## 2. 方案对比 (Original vs Optimized)

| 维度 | 原方案 (Baseline) | 优化方案 (v2) | 差异点分析 |
| :--- | :--- | :--- | :--- |
| **特征工程** | 基础日期解析、简单分箱 | **Target Encoding (K-Fold)**<br>**统计聚合特征**<br>**交互特征** | 引入了强特征 target encoding，挖掘了类别与价格的深层关系；增加了高阶交互特征。 |
| **模型选择** | 单 LightGBM | **LightGBM + XGBoost + CatBoost** | 引入了差异化模型，CatBoost 对类别特征处理更佳，XGBoost 提供互补视角。 |
| **训练策略** | 5-Fold CV | **5-Fold CV + 加权融合** | 简单的平均融合改为基于验证集表现的加权融合。 |
| **异常值处理**| 简单截断 / 剔除 | **基于中位数的平滑处理** | 对 Power 等长尾分布特征处理更稳健，减少信息损失。 |
| **参数调优** | 默认/简单参数 | **精细化调优** | 针对过拟合问题，增加了正则化 (`reg_alpha/lambda=1.0`) 并调整了模型复杂度 (`leaves=127`). |

## 3. 详细优化内容

### 3.1 特征工程增强
在 `code/train_optimized.py` 中实现了以下核心特征：

1.  **Target Encoding (目标编码)**:
    -   对 `brand`, `model`, `regionCode` 等高基数特征，计算其对应的 `price` 统计量（均值、中位数、标准差）。
    -   关键点：使用了 **K-Fold 交叉统计** 策略，严格防止标签泄漏（Data Leakage）。
    -   *贡献*: 这是本次提分最大的单一改动。

2.  **统计聚合特征**:
    -   计算分组统计量，如：`brand` 分组下的 `power` 均值、`model` 分组下的 `kilometer` 标准差等。
    -   反映了特定品牌/车型的平均配置水平。

3.  **交互特征**:
    -   构建了 `km_per_year` (年均行驶里程)、`power_per_km` 等具有物理意义的比率特征。
    -   构建了 `brand_bodyType` 等组合类别特征。

### 3.2 模型融合策略
我们训练了三个基模型，并进行了加权融合：

-   **LightGBM**: 训练速度快，精度高。CV MAE ~480。
-   **XGBoost**: 传统的强模型，稳定性好。CV MAE ~484。
-   **CatBoost**: 对类别特征支持最好，无需 One-Hot 编码。CV MAE ~478 (单模型最佳)。

**融合方式**:
```python
Final Prediction = w1 * LGB + w2 * XGB + w3 * CatBoost
```
通过网格搜索确定了最优权重，充分利用了不同模型的优势。

## 4. 复现指南

所有优化代码已整合至 `code/train_optimized.py`。

### 运行环境
需要安装额外的 boosting库：
```bash
pip install lightgbm xgboost catboost
```

### 训练与预测
```bash
cd code
python train_optimized.py
```
脚本将自动完成：
1.  特征工程构建
2.  三模型训练 (5-Fold CV)
3.  生成预测文件 `prediction_result/predictions_testB_v2.csv`
