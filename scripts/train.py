"""
完整的训练脚本 - 特征工程 + 模型训练 + 生成提交文件
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings("ignore")

# 配置
RANDOM_SEED = 42
N_FOLDS = 5

print("=" * 60)
print("天池二手车价格预测 - 完整训练流程")
print("=" * 60)

# ============== 1. 加载数据 ==============
print("\n[1/6] 加载数据...")
train = pd.read_csv("data/raw/used_car_train_20200313.csv", sep=" ")
test = pd.read_csv("data/raw/used_car_testA_20200313.csv", sep=" ")
print(f"训练集: {train.shape}, 测试集: {test.shape}")

# ============== 2. 特征工程 ==============
print("\n[2/6] 特征工程...")

def feature_engineering(df):
    """特征工程 Pipeline"""
    df = df.copy()
    
    # 2.1 处理 notRepairedDamage 的特殊值
    df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
    df["notRepairedDamage"] = df["notRepairedDamage"].astype(float)
    
    # 2.2 处理 power 异常值
    df["power"] = df["power"].clip(upper=600)
    
    # 2.3 解析日期特征
    df["regDate"] = df["regDate"].astype(str)
    df["creatDate"] = df["creatDate"].astype(str)
    
    df["regDate_year"] = df["regDate"].str[:4].astype(int)
    df["regDate_month"] = df["regDate"].str[4:6].astype(int)
    df["creatDate_year"] = df["creatDate"].str[:4].astype(int)
    df["creatDate_month"] = df["creatDate"].str[4:6].astype(int)
    
    # 2.4 计算车龄
    df["car_age_year"] = df["creatDate_year"] - df["regDate_year"]
    df["car_age_month"] = (df["creatDate_year"] - df["regDate_year"]) * 12 + \
                          (df["creatDate_month"] - df["regDate_month"])
    df["car_age_year"] = df["car_age_year"].clip(lower=0)
    df["car_age_month"] = df["car_age_month"].clip(lower=0)
    
    # 2.5 Power 分箱
    df["power_bin"] = pd.cut(df["power"], bins=[0, 50, 100, 150, 200, 300, 600],
                             labels=[0, 1, 2, 3, 4, 5])
    df["power_bin"] = df["power_bin"].astype(float).fillna(-1).astype(int)
    
    # 2.6 Kilometer 分箱
    df["kilometer_bin"] = pd.cut(df["kilometer"], bins=[0, 3, 6, 9, 12, 15, 20],
                                  labels=[0, 1, 2, 3, 4, 5])
    df["kilometer_bin"] = df["kilometer_bin"].astype(float).fillna(-1).astype(int)
    
    # 2.7 每年行驶公里数
    df["km_per_year"] = df["kilometer"] / (df["car_age_year"] + 1)
    
    return df

# 应用特征工程
train = feature_engineering(train)
test = feature_engineering(test)
print("特征工程完成")

# ============== 3. 准备特征 ==============
print("\n[3/6] 准备特征...")

# 删除不需要的列
drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType"]
feature_cols = [col for col in train.columns if col not in drop_cols + ["price"]]

# 标签编码
le_dict = {}
for col in feature_cols:
    if train[col].dtype == "object":
        le = LabelEncoder()
        train[col] = train[col].fillna("unknown")
        test[col] = test[col].fillna("unknown")
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        le_dict[col] = le

# 填充缺失值
for col in feature_cols:
    if train[col].isnull().any():
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

X = train[feature_cols]
y = train["price"]
X_test = test[feature_cols]

print(f"特征数量: {len(feature_cols)}")
print(f"训练集: {X.shape}, 测试集: {X_test.shape}")

# ============== 4. LightGBM 训练 ==============
print("\n[4/6] LightGBM 5-Fold 训练...")

lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "verbose": -1
}

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# 对数变换
y_log = np.log1p(y)

for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y_log.iloc[train_idx], y_log.iloc[valid_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=500)
        ]
    )
    
    # 验证集预测
    val_pred = np.expm1(model.predict(X_valid))
    val_pred = np.clip(val_pred, 0, None)
    oof_preds[valid_idx] = val_pred
    
    # 测试集预测
    test_pred = np.expm1(model.predict(X_test))
    test_pred = np.clip(test_pred, 0, None)
    test_preds += test_pred / N_FOLDS
    
    fold_mae = mean_absolute_error(y.iloc[valid_idx], val_pred)
    print(f"Fold {fold + 1} MAE: {fold_mae:.4f}")

# ============== 5. 评估结果 ==============
print("\n[5/6] 评估结果...")
overall_mae = mean_absolute_error(y, oof_preds)
print(f"Overall CV MAE: {overall_mae:.4f}")

# ============== 6. 生成提交文件 ==============
print("\n[6/6] 生成提交文件...")
os.makedirs("submissions", exist_ok=True)

submission = pd.DataFrame({
    "SaleID": test["SaleID"],
    "price": test_preds
})
submission["price"] = submission["price"].clip(lower=0)

submission.to_csv("submissions/submit_lgb.csv", index=False)
print(f"提交文件已保存: submissions/submit_lgb.csv")

# 保存特征重要性
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 10 特征重要性:")
print(importance.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
