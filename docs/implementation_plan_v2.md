# 二手车价格预测模型优化方案

**当前 MAE**: 497 → **目标 MAE**: < 410（进入前500名）

## 瓶颈分析

当前模型的主要问题：
1. **特征工程太简单** — 只有基础的日期、分箱、比率特征，缺少 target encoding 和统计聚合特征
2. **只用了 LightGBM 单模型** — 缺少多模型融合
3. **高基数特征处理不当** — `name`(99K), `regionCode`(7905) 直接 LabelEncoding 效果差
4. **缺失值处理粗糙** — bodyType(4506)、fuelType(8680)、gearbox(5981) 缺失未精细处理

## Proposed Changes

### 特征工程增强

#### [NEW] [train_optimized.py](file:///Users/tiandee/TianProjects/solis-ai/code/train_optimized.py)

一体化的优化训练+预测脚本，包含以下增强：

**1. Target Encoding（最大提升点）**

用训练集中各分组的 `price` 统计量作为特征，用 KFold 防泄漏：
- `brand` → 均价、中位价、标准差
- [model](file:///Users/tiandee/TianProjects/solis-ai/model/train_model.py#119-127) → 均价、中位价、标准差
- `bodyType` / `fuelType` → 均价
- `regionCode` → 均价

**2. 更多统计聚合特征**
- `brand` × price 的 max/min/count
- [model](file:///Users/tiandee/TianProjects/solis-ai/model/train_model.py#119-127) × power 的均值
- `brand` × `bodyType` 交叉组合下的均价

**3. 增强日期特征**
- 使用天数精确计算车龄（而非年/月）
- 车龄与 power/kilometer 的交互比率

**4. 异常值处理优化**
- price 极端值 clip (1~99999)
- power 极端值用中位数替代而非截断

**5. 多模型训练与融合**

| 模型 | 预期权重 | 说明 |
|------|---------|------|
| LightGBM | 0.4 | 更多 boost rounds (10000), 精调参数 |
| XGBoost | 0.3 | 互补 LGB 的决策边界 |
| CatBoost | 0.3 | 天然处理类别特征，天然处理缺失值 |

加权融合三个模型的预测结果。

## Verification Plan

### Automated Tests

运行训练脚本，观察 5-Fold CV MAE 是否降到 400 以下：

```bash
source ~/miniconda3/bin/activate && cd /Users/tiandee/TianProjects/solis-ai/code && python train_optimized.py 2>&1
```

输出将包含每折的 MAE 和总体 CV MAE，期望 CV MAE < 400。

### Manual Verification

将生成的 `prediction_result/predictions_testB_v2.csv` 提交到天池长期赛，确认线上分数 < 410。
