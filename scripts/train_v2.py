"""
改进版训练脚本 - 增加统计特征
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

RANDOM_SEED = 42
N_FOLDS = 5

print("=" * 60)
print("天池二手车价格预测 - 改进版 V2")
print("=" * 60)

# 1. 加载数据
print("\n[1/6] 加载数据...")
train = pd.read_csv("data/raw/used_car_train_20200313.csv", sep=" ")
test = pd.read_csv("data/raw/used_car_testA_20200313.csv", sep=" ")
print(f"训练集: {train.shape}, 测试集: {test.shape}")

# 合并处理
train["is_train"] = 1
test["is_train"] = 0
test["price"] = -1
data = pd.concat([train, test], ignore_index=True)

# 2. 基础特征工程
print("\n[2/6] 基础特征工程...")

# 处理特殊值
data["notRepairedDamage"] = data["notRepairedDamage"].replace("-", np.nan).astype(float)

# 处理异常值
data["power"] = data["power"].clip(lower=0, upper=600)

# 日期特征
data["regDate"] = data["regDate"].astype(str)
data["creatDate"] = data["creatDate"].astype(str)
data["regDate_year"] = data["regDate"].str[:4].astype(int)
data["regDate_month"] = data["regDate"].str[4:6].astype(int)
data["creatDate_year"] = data["creatDate"].str[:4].astype(int)
data["creatDate_month"] = data["creatDate"].str[4:6].astype(int)

# 车龄
data["car_age_days"] = (pd.to_datetime(data["creatDate"], format="%Y%m%d", errors="coerce") - 
                        pd.to_datetime(data["regDate"], format="%Y%m%d", errors="coerce")).dt.days
data["car_age_days"] = data["car_age_days"].fillna(0).clip(lower=0)
data["car_age_year"] = data["car_age_days"] / 365

# 分箱
data["power_bin"] = pd.cut(data["power"], bins=[-1, 50, 100, 150, 200, 300, 600], labels=False)
data["kilometer_bin"] = pd.cut(data["kilometer"], bins=[-1, 3, 6, 9, 12, 15, 20], labels=False)

# 比率特征
data["km_per_year"] = data["kilometer"] / (data["car_age_year"] + 1)
data["power_per_v"] = data["power"] / (data["v_0"].abs() + 1)

# 3. 统计特征（安全特征 - 不使用目标变量）
print("\n[3/6] 统计特征...")

# 只使用 count 类特征（无目标泄露）
for col in ["brand", "model", "bodyType", "fuelType", "regionCode"]:
    col_count = data.groupby(col).size()
    data[f"{col}_count"] = data[col].map(col_count)
    data[f"{col}_count"] = data[f"{col}_count"].fillna(0)
    
    # power 统计 (安全 - 非目标变量)
    col_power_mean = data.groupby(col)["power"].mean()
    col_power_std = data.groupby(col)["power"].std()
    data[f"{col}_power_mean"] = data[col].map(col_power_mean)
    data[f"{col}_power_std"] = data[col].map(col_power_std)
    
    # kilometer 统计 (安全)
    col_km_mean = data.groupby(col)["kilometer"].mean()
    data[f"{col}_km_mean"] = data[col].map(col_km_mean)
    
    # 填充缺失
    for stat_col in [f"{col}_count", f"{col}_power_mean", f"{col}_power_std", f"{col}_km_mean"]:
        if stat_col in data.columns:
            data[stat_col] = data[stat_col].fillna(data[stat_col].median())

# 交叉特征（不涉及目标变量）
data["brand_model"] = data["brand"].astype(str) + "_" + data["model"].astype(str)
bm_count = data.groupby("brand_model").size()
data["brand_model_count"] = data["brand_model"].map(bm_count)
data["brand_model_count"] = data["brand_model_count"].fillna(0)

print(f"生成统计特征完成 - 已移除价格统计特征以避免目标泄露")

# 4. 准备特征
print("\n[4/6] 准备特征...")

drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType", 
             "is_train", "price", "brand_model"]
feature_cols = [col for col in data.columns if col not in drop_cols]

# 标签编码
for col in feature_cols:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = data[col].fillna("unknown").astype(str)
        data[col] = le.fit_transform(data[col])
    else:
        data[col] = data[col].fillna(data[col].median())

# 分离训练和测试
train_df = data[data["is_train"] == 1].reset_index(drop=True)
test_df = data[data["is_train"] == 0].reset_index(drop=True)

X = train_df[feature_cols]
y = train_df["price"]
X_test = test_df[feature_cols]
test_ids = test_df["SaleID"].values

print(f"特征数量: {len(feature_cols)}")

# 5. 训练
print("\n[5/6] LightGBM 5-Fold 训练...")

lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1
}

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
y_log = np.log1p(y)

for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
    print(f"\nFold {fold + 1}/{N_FOLDS}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y_log.iloc[train_idx], y_log.iloc[valid_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=10000,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    
    val_pred = np.expm1(model.predict(X_valid))
    val_pred = np.clip(val_pred, 0, None)
    oof_preds[valid_idx] = val_pred
    
    test_pred = np.expm1(model.predict(X_test))
    test_pred = np.clip(test_pred, 0, None)
    test_preds += test_pred / N_FOLDS
    
    fold_mae = mean_absolute_error(y.iloc[valid_idx], val_pred)
    print(f"Fold {fold + 1} MAE: {fold_mae:.4f}")

overall_mae = mean_absolute_error(y, oof_preds)
print(f"\nOverall CV MAE: {overall_mae:.4f}")

# 6. 保存
print("\n[6/6] 保存结果...")
os.makedirs("prediction_result", exist_ok=True)

submission = pd.DataFrame({"SaleID": test_ids.astype(int), "price": test_preds})
submission["price"] = submission["price"].clip(lower=0)
submission.to_csv("prediction_result/predictions.csv", index=False)

print(f"保存完成: prediction_result/predictions.csv")
print(f"预测均值: {submission['price'].mean():.2f}")

# 特征重要性
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importance()
}).sort_values("importance", ascending=False)
print("\nTop 15 特征:")
print(importance.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
