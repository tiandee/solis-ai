"""
二手车价格预测 - v7 匿名特征深度挖掘 + 智能融合
目标: 使用 v_0~v_14 深度特征工程 + Scipy 权重优化 + 伪标签

策略 A: 匿名特征工程
  1. v_i * v_j 两两交互 (Top 相关对)
  2. PCA 降维生成主成分
  3. v_i 按 brand/model 分组的统计量
  4. v_i 非线性变换 (平方, log)

策略 C: 智能融合
  1. scipy.optimize 最优权重
  2. 跨版本模型融合
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
import time

from train_optimized import (
    preprocess, create_date_features, create_basic_features,
    target_encode_kfold, create_statistical_features,
    LGB_PARAMS, XGB_PARAMS
)

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_FOLDS = 5

# ===================================================================
# 策略 A: 匿名特征深度挖掘
# ===================================================================

def create_anonymous_features(df):
    """对 v_0~v_14 进行深度特征工程"""
    df = df.copy()
    v_cols = [f"v_{i}" for i in range(15)]
    
    # --- 1. 两两交互特征 (Top 相关对, 手选最有用的) ---
    # 选择方差最大的几个特征做交互，避免特征爆炸
    top_v = ["v_0", "v_1", "v_3", "v_4", "v_5", "v_8", "v_12"]
    for i in range(len(top_v)):
        for j in range(i+1, len(top_v)):
            vi, vj = top_v[i], top_v[j]
            df[f"{vi}_x_{vj}"] = df[vi] * df[vj]
            df[f"{vi}_sub_{vj}"] = df[vi] - df[vj]
    
    # --- 2. 非线性变换 ---
    for col in v_cols:
        df[f"{col}_sq"] = df[col] ** 2
        # 对正值取 log
        df[f"{col}_abs"] = df[col].abs()
    
    # --- 3. 聚合统计 ---
    df["v_mean"] = df[v_cols].mean(axis=1)
    df["v_std"] = df[v_cols].std(axis=1)
    df["v_max"] = df[v_cols].max(axis=1)
    df["v_min"] = df[v_cols].min(axis=1)
    df["v_range"] = df["v_max"] - df["v_min"]
    df["v_skew"] = df[v_cols].skew(axis=1)
    df["v_kurt"] = df[v_cols].kurtosis(axis=1)
    
    # --- 4. 与已知特征的交互 ---
    if "power" in df.columns:
        df["v_0_x_power"] = df["v_0"] * df["power"]
        df["v_3_x_power"] = df["v_3"] * df["power"]
    if "car_age_year" in df.columns:
        df["v_0_x_age"] = df["v_0"] * df["car_age_year"]
        df["v_3_x_age"] = df["v_3"] * df["car_age_year"]
    if "kilometer" in df.columns:
        df["v_0_x_km"] = df["v_0"] * df["kilometer"]
    
    return df


def add_pca_features(train, test, n_components=5):
    """PCA 降维生成新特征"""
    v_cols = [f"v_{i}" for i in range(15)]
    
    scaler = StandardScaler()
    train_v = scaler.fit_transform(train[v_cols].fillna(0))
    test_v = scaler.transform(test[v_cols].fillna(0))
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    train_pca = pca.fit_transform(train_v)
    test_pca = pca.transform(test_v)
    
    for i in range(n_components):
        train[f"pca_{i}"] = train_pca[:, i]
        test[f"pca_{i}"] = test_pca[:, i]
    
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return train, test


def add_v_group_stats(df):
    """匿名特征按 brand/model 分组统计"""
    df = df.copy()
    v_key = ["v_0", "v_3", "v_8"]
    for v in v_key:
        for grp in ["brand", "model"]:
            if grp in df.columns:
                group = df.groupby(grp)[v].agg(["mean", "std"])
                group.columns = [f"{v}_{grp}_mean", f"{v}_{grp}_std"]
                df = df.merge(group, on=grp, how="left")
                df[f"{v}_{grp}_diff"] = df[v] - df[f"{v}_{grp}_mean"]
    return df


# ===================================================================
# 完整 v7 特征工程 Pipeline
# ===================================================================

def build_v7_features(train, test):
    """v7 完整特征工程 = v2 Pipeline + 匿名特征深挖"""
    print(" [1] 基础预处理...")
    train = preprocess(train)
    test = preprocess(test)
    
    print(" [2] 日期特征...")
    train = create_date_features(train)
    test = create_date_features(test)
    
    print(" [3] 基础衍生特征...")
    train = create_basic_features(train)
    test = create_basic_features(test)
    
    print(" [4] Target Encoding...")
    te_configs = [
        ("brand", ["mean", "median", "std"]),
        ("model", ["mean", "median", "std"]),
        ("bodyType", ["mean"]),
        ("fuelType", ["mean"]),
        ("regionCode", ["mean"]),
        ("power_bin", ["mean"]),
        ("kilometer_bin", ["mean"]),
        ("name", ["mean"]),
    ]
    for col, aggs in te_configs:
        if col in train.columns:
            train, test = target_encode_kfold(train, test, col, "price", agg_funcs=aggs)
    
    print(" [5] 交叉 Target Encoding...")
    train["brand_body"] = train["brand"].astype(str) + "_" + train["bodyType"].astype(str)
    test["brand_body"] = test["brand"].astype(str) + "_" + test["bodyType"].astype(str)
    train, test = target_encode_kfold(train, test, "brand_body", "price", agg_funcs=["mean"])
    
    train["brand_fuel"] = train["brand"].astype(str) + "_" + train["fuelType"].astype(str)
    test["brand_fuel"] = test["brand"].astype(str) + "_" + test["fuelType"].astype(str)
    train, test = target_encode_kfold(train, test, "brand_fuel", "price", agg_funcs=["mean"])
    
    print(" [6] 统计特征...")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = create_statistical_features(combined)
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].copy()
    
    # ★ v7 新增: 匿名特征深度工程 ★
    print(" [7] ★ 匿名特征深度挖掘 (v7)...")
    train = create_anonymous_features(train)
    test = create_anonymous_features(test)
    
    print(" [8] ★ PCA 降维 (v7)...")
    train, test = add_pca_features(train, test, n_components=5)
    
    print(" [9] ★ 匿名特征分组统计 (v7)...")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = add_v_group_stats(combined)
    train = combined.iloc[:len(train)].copy()
    test = combined.iloc[len(train):].copy()
    
    return train, test


# ===================================================================
# 模型训练 (含 Pseudo Labels)
# ===================================================================

def train_lgb_v7(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """LightGBM v7 with Pseudo Labels"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"LightGBM (v7) {n_folds}-Fold")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)],
        )
        
        val_pred = np.expm1(model.predict(X_val)).clip(0)
        oof[val_idx] = val_pred
        test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        print(f"  Fold {fold+1} MAE: {mean_absolute_error(y.iloc[val_idx], val_pred):.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ LightGBM (v7) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_xgb_v7(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """XGBoost v7 with Pseudo Labels"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"XGBoost (v7) {n_folds}-Fold")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
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
        dtest = xgb.DMatrix(X_test)
        test_preds += np.expm1(model.predict(dtest)).clip(0) / n_folds
        print(f"  Fold {fold+1} MAE: {mean_absolute_error(y.iloc[val_idx], val_pred):.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ XGBoost (v7) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_catboost_v7(X, y, X_test, y_test_pseudo, cat_features, n_folds=N_FOLDS):
    """CatBoost v7 with Pseudo Labels"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"CatBoost (v7) {n_folds}-Fold")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            loss_function="MAE",  # Use MAE loss for CatBoost (v5 was best with this)
            random_seed=RANDOM_SEED,
            verbose=500,
            early_stopping_rounds=200,
            cat_features=cat_features,
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        val_pred = np.expm1(model.predict(X_val)).clip(0)
        oof[val_idx] = val_pred
        test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        print(f"  Fold {fold+1} MAE: {mean_absolute_error(y.iloc[val_idx], val_pred):.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ CatBoost (v7) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


# ===================================================================
# 策略 C: 智能融合权重优化
# ===================================================================

def optimize_weights(oof_list, y_true, n_models):
    """用 scipy 最优化找最佳融合权重"""
    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()  # Normalize
        blend = sum(w * oof for w, oof in zip(weights, oof_list))
        return mean_absolute_error(y_true, blend)
    
    # 初始权重: 均匀
    x0 = np.ones(n_models) / n_models
    
    # 约束: 权重之和 = 1
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    # 边界: 0~1
    bounds = [(0, 1)] * n_models
    
    result = minimize(objective, x0, method="SLSQP",
                      constraints=constraints, bounds=bounds,
                      options={"maxiter": 1000})
    
    best_weights = result.x
    best_mae = result.fun
    return best_weights, best_mae


# ===================================================================
# 主流程
# ===================================================================

def main():
    print("=" * 60)
    print("二手车价格预测 - v7 匿名特征深度挖掘 + 智能融合")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/7] 加载数据 & v5 伪标签...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    
    # Load v5 Predictions (LB 442.99 - Best)
    pseudo_path = "../prediction_result/predictions_pseudo_v5.csv"
    pseudo_df = pd.read_csv(pseudo_path)
    if not np.array_equal(pseudo_df["SaleID"].values, test["SaleID"].values):
        pseudo_df = pd.DataFrame({"SaleID": test["SaleID"]}).merge(pseudo_df, on="SaleID", how="left")
    y_test_pseudo = pseudo_df["price"].values
    
    # 2. v7 Feature Engineering
    print("\n[2/7] ★ v7 特征工程 Pipeline...")
    train, test = build_v7_features(train, test)
    
    # 3. Prepare Data
    print("\n[3/7] 准备特征...")
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType",
                 "brand_body", "brand_fuel"]
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + ["price"]]
    
    cat_candidates = ["brand", "model", "bodyType", "fuelType", "gearbox",
                      "regionCode", "power_bin", "kilometer_bin"]
    cat_cols_for_cb = [c for c in cat_candidates if c in feature_cols]
    
    for col in feature_cols:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())
            
    for col in feature_cols:
        if train[col].dtype == "object":
            le = LabelEncoder()
            train[col] = train[col].fillna("unknown")
            test[col] = test[col].fillna("unknown")
            combined = pd.concat([train[col], test[col]]).astype(str)
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
    
    X = train[feature_cols]
    y = train["price"]
    X_test = test[feature_cols]
    
    print(f"  特征数量: {len(feature_cols)} (v2: ~66, v7: {len(feature_cols)})")
    
    # 4. Train Models
    print("\n[4/7] 模型训练 (v7 with Pseudo Labels)...")
    lgb_oof, lgb_test, lgb_mae = train_lgb_v7(X, y, X_test, y_test_pseudo, feature_cols)
    xgb_oof, xgb_test, xgb_mae = train_xgb_v7(X, y, X_test, y_test_pseudo, feature_cols)
    
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols_for_cb:
        X_cb[c] = X_cb[c].astype(int)
        X_test_cb[c] = X_test_cb[c].astype(int)
    cb_oof, cb_test, cb_mae = train_catboost_v7(X_cb, y, X_test_cb, y_test_pseudo, cat_features=cat_cols_for_cb)
    
    # 5. Smart Ensemble (Strategy C)
    print("\n[5/7] ★ 智能融合权重优化 (Strategy C)...")
    oof_list = [lgb_oof, xgb_oof, cb_oof]
    test_list = [lgb_test, xgb_test, cb_test]
    model_names = ["LGB", "XGB", "Cat"]
    
    best_weights, best_mae = optimize_weights(oof_list, y, n_models=3)
    print(f"  最优权重: {dict(zip(model_names, [f'{w:.4f}' for w in best_weights]))}")
    print(f"  最优融合 CV MAE: {best_mae:.2f}")
    
    # Apply optimized weights to test
    final_pred = sum(w * pred for w, pred in zip(best_weights, test_list))
    
    # Also try equal-weight for comparison
    equal_pred = (lgb_test + xgb_test + cb_test) / 3
    equal_oof = (lgb_oof + xgb_oof + cb_oof) / 3
    equal_mae = mean_absolute_error(y, equal_oof)
    print(f"  等权融合 CV MAE: {equal_mae:.2f}")
    
    # 6. Save
    print("\n[6/7] 保存结果...")
    os.makedirs("../prediction_result", exist_ok=True)
    
    submission = pd.DataFrame({"SaleID": test["SaleID"], "price": final_pred})
    submission["price"] = submission["price"].clip(lower=0)
    out_path = "../prediction_result/predictions_v7_smart.csv"
    submission.to_csv(out_path, index=False)
    
    # Also save OOF for future blending
    oof_df = pd.DataFrame({
        "lgb_oof": lgb_oof,
        "xgb_oof": xgb_oof,
        "cb_oof": cb_oof,
        "true": y
    })
    oof_df.to_csv("../prediction_result/oof_v7.csv", index=False)
    
    print(f"\nSaved to: {out_path}")
    
    # 7. Summary
    print("\n[7/7] 总结:")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  LGB v7 MAE: {lgb_mae:.2f}")
    print(f"  XGB v7 MAE: {xgb_mae:.2f}")
    print(f"  Cat v7 MAE: {cb_mae:.2f}")
    print(f"  智能融合 MAE: {best_mae:.2f}")
    print(f"  等权融合 MAE: {equal_mae:.2f}")
    print(f"  最优权重: {dict(zip(model_names, [f'{w:.3f}' for w in best_weights]))}")

if __name__ == "__main__":
    main()
