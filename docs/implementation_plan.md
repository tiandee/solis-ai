# 天池二手车价格预测比赛 - 参赛路径规划

## 比赛概述

| 项目 | 内容 |
|------|------|
| **比赛名称** | 【AI入门系列】车市先知：二手车价格预测学习赛 |
| **任务类型** | 回归问题 (预测二手车价格) |
| **评测标准** | MAE (Mean Absolute Error) |
| **目标** | MAE 越低越好 |

---

## 🗓️ 参赛路径规划

### 阶段一：环境准备与数据获取 (Day 1)

#### 1.1 搭建开发环境
- [ ] 安装 Python 3.8+ 环境
- [ ] 安装核心依赖库：
  ```bash
  pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
  ```
- [ ] 准备 Jupyter Notebook 或 VS Code

#### 1.2 下载数据
- [ ] 注册/登录天池账号
- [ ] 下载以下数据文件：
  - `used_car_sample_submit.csv` - 提交样例
  - `used_car_testB_20200421.zip` - 测试集B
  - `used_car_train_20200313.zip` - 训练集 (~15万条)
  - `used_car_testA_20200313.csv.zip` - 测试集A (~5万条)

---

### 阶段二：探索性数据分析 EDA (Day 2-3)

#### 2.1 理解数据字段

| 字段 | 说明 | 类型 |
|------|------|------|
| SaleID | 交易ID，唯一标识 | 标识符 |
| name | 汽车交易名称，已脱敏 | 分类 |
| regDate | 汽车注册日期 (如20160101) | 时间 |
| model | 车型编码，已脱敏 | 分类 |
| brand | 汽车品牌，已脱敏 | 分类 |
| bodyType | 车身类型 (0-7: 豪华轿车/微型车/厢型车等) | 分类 |
| fuelType | 燃油类型 (0-6: 汽油/柴油/天然气等) | 分类 |
| gearbox | 变速箱 (0:自动, 1:手动) | 二分类 |
| power | 发动机功率 | 数值 |
| kilometer | 汽车行驶公里数 (单位:万km) | 数值 |
| notRepairedDamage | 是否有尚未修复的损坏 | 二分类 |
| regionCode | 地区编码 | 分类 |
| seller | 销售方 | 分类 |
| offerType | 报价类型 | 分类 |
| creatDate | 广告发布时间 | 时间 |
| price | **目标变量** - 二手车交易价格 | 数值 |
| v系列 | v_0 ~ v_14 匿名特征 | 数值 |

#### 2.2 EDA 任务清单
- [ ] 加载数据并查看基本信息 (`df.info()`, `df.describe()`)
- [ ] 检查缺失值分布
- [ ] 分析目标变量 `price` 分布 (是否需要对数变换)
- [ ] 分析各特征与 price 的相关性
- [ ] 可视化重要特征

---

### 阶段三：特征工程 (Day 4-6)

#### 3.1 数据清洗
- [ ] 处理缺失值 (填充/删除)
- [ ] 处理异常值 (如 power 异常大的值)
- [ ] 修复 `notRepairedDamage` 中的 `-` 值

#### 3.2 特征构造
- [ ] **时间特征**：
  - 从 `regDate` 提取：注册年、月、日
  - 从 `creatDate` 提取：发布年、月、日
  - 计算车龄 = 发布日期 - 注册日期
- [ ] **交叉特征**：
  - brand × model 组合
  - bodyType × fuelType 组合
- [ ] **统计特征**：
  - 同品牌/车型的平均价格
  - 同地区的平均价格
- [ ] **分箱特征**：
  - power 分箱
  - kilometer 分箱

#### 3.3 特征编码
- [ ] 低基数分类特征：Label Encoding
- [ ] 高基数分类特征：Target Encoding 或 Frequency Encoding

---

### 阶段四：模型训练 (Day 7-10)

#### 4.1 Baseline 模型
```python
# 快速验证的简单模型
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
```
- [ ] 训练简单模型，建立 baseline 分数

#### 4.2 进阶模型（推荐）
```python
# 树模型三剑客
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
```
- [ ] LightGBM 训练与调参
- [ ] XGBoost 训练与调参  
- [ ] CatBoost 训练与调参

#### 4.3 交叉验证
- [ ] 使用 5-Fold 或 10-Fold 交叉验证
- [ ] 对目标变量做对数变换 `log1p(price)`，预测后还原 `expm1(pred)`

---

### 阶段五：模型调优与融合 (Day 11-14)

#### 5.1 超参数调优
- [ ] 使用 Optuna 或 GridSearchCV 调参
- [ ] 关注关键参数：
  - `learning_rate`
  - `max_depth`
  - `num_leaves`
  - `reg_alpha`, `reg_lambda`

#### 5.2 模型融合
- [ ] **加权平均**：多模型预测结果加权
- [ ] **Stacking**：使用元模型融合
- [ ] **Blending**：简单混合

---

### 阶段六：结果提交 (持续)

#### 6.1 生成提交文件
```python
submit = pd.DataFrame()
submit['SaleID'] = test['SaleID']
submit['price'] = predictions
submit.to_csv('submit.csv', index=False)
```

#### 6.2 提交格式
```
SaleID,price
10000,7500
100001,7888
...
```

---

## 📁 建议项目结构

```
solis-ai/
├── data/                      # 数据文件
│   ├── used_car_train.csv
│   ├── used_car_testA.csv
│   └── used_car_testB.csv
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # 探索性分析
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                       # 源代码
│   ├── features.py           # 特征工程
│   ├── models.py             # 模型定义
│   └── utils.py              # 工具函数
├── submissions/               # 提交文件
└── requirements.txt
```

---

## 🎯 快速上分技巧

1. **对数变换目标变量** - price 分布偏斜，用 `log1p()` 变换效果显著
2. **车龄是重要特征** - 从日期计算出的车龄与价格强相关
3. **处理异常 power** - 部分 power 值超过 600，建议截断或删除
4. **匿名特征 v_0~v_14 很重要** - 不要忽略这些特征
5. **LightGBM 通常是最佳选择** - 速度快、效果好

---

## ⏱️ 预估时间表

| 阶段 | 天数 | 预期成果 |
|------|------|----------|
| 环境准备 | 1天 | 环境就绪，数据下载 |
| EDA | 2天 | 理解数据，发现问题 |
| 特征工程 | 3天 | 构建优质特征 |
| 模型训练 | 4天 | 单模型调优 |
| 模型融合 | 3天 | 最终提交 |
| **总计** | **~2周** | |

---

## 下一步行动

请确认：
1. 你是否已有 Python 开发环境？
2. 是否需要我帮你搭建项目结构和初始代码？
3. 你的时间安排是怎样的（全职还是业余时间）？
