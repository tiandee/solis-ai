"""
二手车价格预测 - 优化版本
目标: MAE < 400 (当前 ~497)

改进点:
1. Target Encoding (KFold防泄漏)
2. 更多统计聚合与交互特征
3. LightGBM + XGBoost + CatBoost 三模型融合
4. 更精细的异常值与缺失值处理
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
import time

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_FOLDS = 5

# ===================================================================
# 特征工程
# ===================================================================

def preprocess(df):
    """基础预处理"""
    df = df.copy()
    # notRepairedDamage 特殊值处理
    df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan).astype(float)
    # power 异常值: 用中位数替代极端值(而非简单截断)
    power_median = df["power"].median()
    df.loc[df["power"] > 600, "power"] = power_median
    df.loc[df["power"] <= 0, "power"] = power_median
    return df


def create_date_features(df):
    """日期特征 - 增强版"""
    df = df.copy()
    
    df["regDate"] = df["regDate"].astype(str)
    df["creatDate"] = df["creatDate"].astype(str)
    
    # 基础日期解析
    df["regDate_year"] = df["regDate"].str[:4].astype(int)
    df["regDate_month"] = df["regDate"].str[4:6].astype(int)
    df["creatDate_year"] = df["creatDate"].str[:4].astype(int)
    df["creatDate_month"] = df["creatDate"].str[4:6].astype(int)
    
    # 车龄(年和月)
    df["car_age_year"] = (df["creatDate_year"] - df["regDate_year"]).clip(lower=0)
    df["car_age_month"] = (
        (df["creatDate_year"] - df["regDate_year"]) * 12 +
        (df["creatDate_month"] - df["regDate_month"])
    ).clip(lower=0)
    
    # 车龄精细化: 用天数近似
    try:
        reg_date = pd.to_datetime(df["regDate"], format="%Y%m%d", errors="coerce")
        creat_date = pd.to_datetime(df["creatDate"], format="%Y%m%d", errors="coerce")
        df["car_age_days"] = (creat_date - reg_date).dt.days.fillna(0).clip(lower=0)
    except:
        df["car_age_days"] = df["car_age_month"] * 30
    
    return df


def create_basic_features(df):
    """基础衍生特征"""
    df = df.copy()
    
    # 分箱特征
    df["power_bin"] = pd.cut(df["power"], bins=[0, 50, 100, 150, 200, 300, 600],
                             labels=[0, 1, 2, 3, 4, 5])
    df["power_bin"] = df["power_bin"].astype(float).fillna(-1).astype(int)
    
    df["kilometer_bin"] = pd.cut(df["kilometer"], bins=[0, 3, 6, 9, 12, 15, 20],
                                  labels=[0, 1, 2, 3, 4, 5])
    df["kilometer_bin"] = df["kilometer_bin"].astype(float).fillna(-1).astype(int)
    
    # 比率特征
    df["km_per_year"] = df["kilometer"] / (df["car_age_year"] + 1)
    df["km_per_month"] = df["kilometer"] / (df["car_age_month"] + 1)
    
    # power 相关
    df["power_per_km"] = df["power"] / (df["kilometer"] + 1)
    
    # 车龄 × power 交互
    df["age_power"] = df["car_age_year"] * df["power"]
    df["age_km"] = df["car_age_year"] * df["kilometer"]
    
    return df


def target_encode_kfold(train_df, test_df, col, target, n_folds=5, 
                         agg_funcs=["mean"], smooth=100):
    """
    KFold Target Encoding 防止数据泄漏
    smooth: 平滑参数,防止小样本过拟合
    """
    global_mean = train_df[target].mean()
    
    for agg in agg_funcs:
        new_col = f"{col}_target_{agg}"
        train_df[new_col] = np.nan
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        for tr_idx, val_idx in kf.split(train_df):
            tr_data = train_df.iloc[tr_idx]
            
            if agg == "mean":
                agg_vals = tr_data.groupby(col)[target].agg(["mean", "count"])
                # 平滑处理
                agg_vals["smoothed"] = (
                    (agg_vals["mean"] * agg_vals["count"] + global_mean * smooth) / 
                    (agg_vals["count"] + smooth)
                )
                mapping = agg_vals["smoothed"]
            elif agg == "median":
                mapping = tr_data.groupby(col)[target].median()
            elif agg == "std":
                mapping = tr_data.groupby(col)[target].std()
            elif agg == "count":
                mapping = tr_data.groupby(col)[target].count()
            elif agg == "max":
                mapping = tr_data.groupby(col)[target].max()
            elif agg == "min":
                mapping = tr_data.groupby(col)[target].min()
            
            train_df.loc[train_df.index[val_idx], new_col] = \
                train_df.iloc[val_idx][col].map(mapping)
        
        # 填充缺失
        train_df[new_col] = train_df[new_col].fillna(global_mean)
        
        # 测试集: 用全量训练集的统计量
        if agg == "mean":
            full_agg = train_df.groupby(col)[target].agg(["mean", "count"])
            full_agg["smoothed"] = (
                (full_agg["mean"] * full_agg["count"] + global_mean * smooth) / 
                (full_agg["count"] + smooth)
            )
            test_mapping = full_agg["smoothed"]
        elif agg == "median":
            test_mapping = train_df.groupby(col)[target].median()
        elif agg == "std":
            test_mapping = train_df.groupby(col)[target].std()
        elif agg == "count":
            test_mapping = train_df.groupby(col)[target].count()
        elif agg == "max":
            test_mapping = train_df.groupby(col)[target].max()
        elif agg == "min":
            test_mapping = train_df.groupby(col)[target].min()
        
        test_df[new_col] = test_df[col].map(test_mapping).fillna(global_mean)
    
    return train_df, test_df


def create_statistical_features(df, is_train=True, stats_cache=None):
    """基于分组的统计特征 (非target, 不会泄漏)"""
    df = df.copy()
    
    # 品牌维度的 power 统计
    for col in ["brand", "model"]:
        if col in df.columns:
            group = df.groupby(col)["power"].agg(["mean", "std", "max", "min"])
            group.columns = [f"{col}_power_{s}" for s in ["mean", "std", "max", "min"]]
            df = df.merge(group, on=col, how="left")
    
    # 品牌维度的 kilometer 统计
    for col in ["brand"]:
        group = df.groupby(col)["kilometer"].agg(["mean", "std"])
        group.columns = [f"{col}_km_mean", f"{col}_km_std"]
        df = df.merge(group, on=col, how="left")
    
    # 计数特征
    for col in ["brand", "model", "regionCode", "bodyType"]:
        if col in df.columns:
            cnt = df.groupby(col).size().reset_index(name=f"{col}_count")
            df = df.merge(cnt, on=col, how="left")
    
    return df


def build_all_features(train, test):
    """完整特征工程 Pipeline"""
    print("  [1] 基础预处理...")
    train = preprocess(train)
    test = preprocess(test)
    
    print("  [2] 日期特征...")
    train = create_date_features(train)
    test = create_date_features(test)
    
    print("  [3] 基础衍生特征...")
    train = create_basic_features(train)
    test = create_basic_features(test)
    
    print("  [4] Target Encoding...")
    # 主要分类特征的 target encoding
    te_configs = [
        ("brand", ["mean", "median", "std"]),
        ("model", ["mean", "median", "std"]),
        ("bodyType", ["mean"]),
        ("fuelType", ["mean"]),
        ("regionCode", ["mean"]),
        ("power_bin", ["mean"]),
        ("kilometer_bin", ["mean"]),
        ("name", ["mean"]),  # 高基数, 用平滑
    ]
    
    for col, aggs in te_configs:
        if col in train.columns:
            train, test = target_encode_kfold(train, test, col, "price", agg_funcs=aggs)
    
    # 交叉 target encoding
    print("  [5] 交叉 Target Encoding...")
    train["brand_body"] = train["brand"].astype(str) + "_" + train["bodyType"].astype(str)
    test["brand_body"] = test["brand"].astype(str) + "_" + test["bodyType"].astype(str)
    train, test = target_encode_kfold(train, test, "brand_body", "price", agg_funcs=["mean"])
    
    train["brand_fuel"] = train["brand"].astype(str) + "_" + train["fuelType"].astype(str)
    test["brand_fuel"] = test["brand"].astype(str) + "_" + test["fuelType"].astype(str)
    train, test = target_encode_kfold(train, test, "brand_fuel", "price", agg_funcs=["mean"])
    
    print("  [6] 统计特征...")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = create_statistical_features(combined)
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].copy()
    
    return train, test


# ===================================================================
# 模型训练
# ===================================================================

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "learning_rate": 0.02,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
    "nthread": -1,
}


def train_lgb(X, y, X_test, feature_cols, n_folds=N_FOLDS):
    """LightGBM KFold 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    y_log = np.log1p(y)
    
    print(f"\n{'='*50}")
    print(f"LightGBM {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(500),
            ],
        )
        
        val_pred = np.expm1(model.predict(X_val)).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ LightGBM CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_xgb(X, y, X_test, feature_cols, n_folds=N_FOLDS):
    """XGBoost KFold 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    y_log = np.log1p(y)
    
    print(f"\n{'='*50}")
    print(f"XGBoost {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=10000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=500,
        )
        
        val_pred = np.expm1(model.predict(dvalid)).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            dtest = xgb.DMatrix(X_test)
            test_preds += np.expm1(model.predict(dtest)).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ XGBoost CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_catboost(X, y, X_test, cat_features, n_folds=N_FOLDS):
    """CatBoost KFold 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    y_log = np.log1p(y)
    
    print(f"\n{'='*50}")
    print(f"CatBoost {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            random_seed=RANDOM_SEED,
            verbose=500,
            early_stopping_rounds=200,
            eval_metric="MAE",
            cat_features=cat_features,
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        val_pred = np.expm1(model.predict(X_val)).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ CatBoost CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def find_best_weights(oof_list, y_true, step=0.05):
    """网格搜索最优融合权重"""
    best_mae = float("inf")
    best_w = None
    
    n_models = len(oof_list)
    if n_models == 3:
        for w1 in np.arange(0.1, 0.8, step):
            for w2 in np.arange(0.1, 0.8 - w1 + step, step):
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                blend = w1 * oof_list[0] + w2 * oof_list[1] + w3 * oof_list[2]
                mae = mean_absolute_error(y_true, blend)
                if mae < best_mae:
                    best_mae = mae
                    best_w = [w1, w2, w3]
    elif n_models == 2:
        for w1 in np.arange(0.1, 0.9, step):
            w2 = 1.0 - w1
            blend = w1 * oof_list[0] + w2 * oof_list[1]
            mae = mean_absolute_error(y_true, blend)
            if mae < best_mae:
                best_mae = mae
                best_w = [w1, w2]
    
    return best_w, best_mae


# ===================================================================
# 主流程
# ===================================================================

def main():
    start_time = time.time()
    
    print("=" * 60)
    print("二手车价格预测 - 优化版 v2")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    print(f"训练集: {train.shape}, 测试集B: {test.shape}")
    
    test_ids = test["SaleID"].values
    
    # 2. 特征工程
    print("\n[2/6] 特征工程...")
    train, test = build_all_features(train, test)
    
    # 3. 准备特征
    print("\n[3/6] 准备特征...")
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType",
                 "brand_body", "brand_fuel"]
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + ["price"]]
    
    # 识别类别特征 (for CatBoost)
    cat_cols_for_cb = []
    for col in feature_cols:
        if train[col].dtype == "object":
            le = LabelEncoder()
            train[col] = train[col].fillna("unknown")
            test[col] = test[col].fillna("unknown")
            combined = pd.concat([train[col], test[col]])
            le.fit(combined)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
    
    # 指定 CatBoost 类别特征 (整数型分类)
    cat_candidates = ["brand", "model", "bodyType", "fuelType", "gearbox",
                      "regionCode", "power_bin", "kilometer_bin"]
    cat_cols_for_cb = [c for c in cat_candidates if c in feature_cols]
    
    # 填充缺失值
    for col in feature_cols:
        if train[col].isnull().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
        if test[col].isnull().any():
            median_val = train[col].median()
            test[col] = test[col].fillna(median_val)
    
    X = train[feature_cols]
    y = train["price"]
    X_test = test[feature_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"CatBoost 类别特征: {cat_cols_for_cb}")
    
    # 4. 模型训练
    print("\n[4/6] 模型训练...")
    
    lgb_oof, lgb_test, lgb_mae = train_lgb(X, y, X_test, feature_cols)
    xgb_oof, xgb_test, xgb_mae = train_xgb(X, y, X_test, feature_cols)
    # xgb_mae = 9999
    # xgb_oof = np.zeros_like(lgb_oof)
    # xgb_test = np.zeros_like(lgb_test)
    
    # CatBoost: 需要 int 型类别特征
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols_for_cb:
        X_cb[c] = X_cb[c].astype(int)
        X_test_cb[c] = X_test_cb[c].astype(int)
    
    cb_oof, cb_test, cb_mae = train_catboost(
        X_cb, y, X_test_cb, cat_features=cat_cols_for_cb
    )
    # cb_mae = 9999
    # cb_oof = np.zeros_like(lgb_oof)
    # cb_test = np.zeros_like(lgb_test)
    
    # 5. 模型融合
    print("\n[5/6] 模型融合...")
    
    oof_list = [lgb_oof, xgb_oof, cb_oof]
    test_list = [lgb_test, xgb_test, cb_test]
    
    # 搜索最优权重
    best_weights, best_mae = find_best_weights(oof_list, y)
    print(f"最优权重: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
    print(f"★ 融合 CV MAE: {best_mae:.2f}")
    
    # 生成融合预测
    final_pred = sum(w * p for w, p in zip(best_weights, test_list))
    
    # final_pred = lgb_test
    # best_mae = lgb_mae
    
    # 6. 保存结果
    print("\n[6/6] 保存结果...")
    os.makedirs("../prediction_result", exist_ok=True)
    
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": final_pred
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    output_path = "../prediction_result/predictions_testB_v2.csv"
    submission.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"训练完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    print(f"\n各模型 CV MAE:")
    print(f"  LightGBM:  {lgb_mae:.2f}")
    print(f"  XGBoost:   {xgb_mae:.2f}")
    print(f"  CatBoost:  {cb_mae:.2f}")
    print(f"  融合:      {best_mae:.2f}")

    # Save OOF and Test Preds for Stacking
    print("\n[7/6] 保存 Stacking 所需文件...")
    # OOF
    pd.DataFrame({"SaleID": train["SaleID"], "price": lgb_oof}).to_csv("../prediction_result/oof_lgb.csv", index=False)
    pd.DataFrame({"SaleID": train["SaleID"], "price": xgb_oof}).to_csv("../prediction_result/oof_xgb.csv", index=False)
    pd.DataFrame({"SaleID": train["SaleID"], "price": cb_oof}).to_csv("../prediction_result/oof_cat.csv", index=False)
    
    # Test Preds
    pd.DataFrame({"SaleID": test_ids, "price": lgb_test}).to_csv("../prediction_result/pred_lgb.csv", index=False)
    pd.DataFrame({"SaleID": test_ids, "price": xgb_test}).to_csv("../prediction_result/pred_xgb.csv", index=False)
    pd.DataFrame({"SaleID": test_ids, "price": cb_test}).to_csv("../prediction_result/pred_cat.csv", index=False)

    print(f"\n预测结果: {output_path}")
    print(f"样本数: {len(submission)}")
    print(f"价格均值: {submission['price'].mean():.2f}")
    print(f"价格范围: {submission['price'].min():.2f} ~ {submission['price'].max():.2f}")


if __name__ == "__main__":
    main()
