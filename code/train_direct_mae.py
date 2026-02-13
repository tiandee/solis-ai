"""
二手车价格预测 - v4 Direct MAE 优化
目标: MAE < 400 (通过直接优化 MAE 损失函数)

策略:
1. 不再使用 Log 变换，直接预测原始价格。
2. 使用 L1 (MAE)Loss 作为目标函数。
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

# 复用优化版的特征工程
try:
    from train_optimized import build_all_features
except ImportError:
    # Handle running from different dir
    pass

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_FOLDS = 5

# ===================================================================
# 模型训练 (MAE 版)
# ===================================================================

LGB_PARAMS_MAE = {
    "objective": "regression_l1",  # MAE Loss
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 10.0,
    "reg_lambda": 10.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS_MAE = {
    "objective": "reg:absoluteerror", # MAE Loss
    "eval_metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 5.0,
    "reg_lambda": 5.0,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
    "nthread": -1,
}


def train_lgb_mae(X, y, X_test, feature_cols, n_folds=N_FOLDS):
    """LightGBM KFold 训练 (Direct MAE)"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    # NO Log Transform
    # y = np.log1p(y) 
    
    print(f"\n{'='*50}")
    print(f"LightGBM (MAE) {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            LGB_PARAMS_MAE, dtrain,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(500),
            ],
        )
        
        # NO expm1
        val_pred = model.predict(X_val).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            test_preds += model.predict(X_test).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ LightGBM (MAE) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_xgb_mae(X, y, X_test, feature_cols, n_folds=N_FOLDS):
    """XGBoost KFold 训练 (Direct MAE)"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    print(f"\n{'='*50}")
    print(f"XGBoost (MAE) {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            XGB_PARAMS_MAE, dtrain,
            num_boost_round=10000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=500,
        )
        
        val_pred = model.predict(dvalid).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            dtest = xgb.DMatrix(X_test)
            test_preds += model.predict(dtest).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ XGBoost (MAE) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_catboost_mae(X, y, X_test, cat_features, n_folds=N_FOLDS):
    """CatBoost KFold 训练 (Direct MAE)"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    
    print(f"\n{'='*50}")
    print(f"CatBoost (MAE) {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(
            learning_rate=0.05,
            depth=6,
            loss_function='MAE', # MAE Loss
            eval_metric='MAE',
            l2_leaf_reg=10,
            random_seed=RANDOM_SEED,
            verbose=500,
            early_stopping_rounds=200,
            cat_features=cat_features,
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        val_pred = model.predict(X_val).clip(0)
        oof[val_idx] = val_pred
        
        if X_test is not None:
            test_preds += model.predict(X_test).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ CatBoost (MAE) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("二手车价格预测 - v4 Direct MAE 优化")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    test["price"] = -1 # Placeholder
    
    test_ids = test["SaleID"].values
    
    # 2. 特征工程 (复用)
    print("\n[2/6] 特征工程...")
    # 这里需要确保 build_all_features 可导入
    # 简单起见，我们假设它在同一个包下，或者直接把 build_all_features 及其依赖复制过来？
    # 为了避免代码重复，我们尝试导入
    try:
        from train_optimized import build_all_features
    except ImportError:
        print("Error: train_optimized.py not found or failed to import.")
        return

    train, test = build_all_features(train, test)
    
    # 3. 准备特征
    print("\n[3/6] 准备特征...")
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType",
                 "brand_body", "brand_fuel"]
    feature_cols = [col for col in train.columns 
                    if col not in drop_cols + ["price"]]
    
    # CatBoost 类别
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
            
    # Label Encode object cols
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
    
    # 4. 模型训练
    print("\n[4/6] 模型训练 (Direct MAE)...")
    
    # 优先训练 LightGBM 进行验证
    lgb_oof, lgb_test, lgb_mae = train_lgb_mae(X, y, X_test, feature_cols)
    
    # 如果 LGB 效果好 (< 450), 再开启其他模型
    xgb_oof, xgb_test, xgb_mae = train_xgb_mae(X, y, X_test, feature_cols)
    # xgb_mae = 9999
    # xgb_test = np.zeros_like(lgb_test)
    
    # CatBoost
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols_for_cb:
        X_cb[c] = X_cb[c].astype(int)
        X_test_cb[c] = X_test_cb[c].astype(int)
        
    cb_oof, cb_test, cb_mae = train_catboost_mae(X_cb, y, X_test_cb, cat_features=cat_cols_for_cb)
    # cb_mae = 9999
    # cb_test = np.zeros_like(lgb_test)
    
    # 5. 简单融合
    print("\n[5/6] 简单平均融合...")
    final_pred = (lgb_test + xgb_test + cb_test) / 3
    # final_pred = lgb_test
    
    # 6. 保存
    print("\n[6/6] 保存结果...")
    os.makedirs("../prediction_result", exist_ok=True)
    
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": final_pred
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    out_path = "../prediction_result/predictions_mae_v4.csv"
    submission.to_csv(out_path, index=False)
    
    print(f"\nSaved to: {out_path}")
    print(f"LGB MAE: {lgb_mae:.2f}")
    if xgb_mae != 9999: print(f"XGB MAE: {xgb_mae:.2f}")
    if cb_mae != 9999: print(f"Cat MAE: {cb_mae:.2f}")

if __name__ == "__main__":
    main()
