"""
二手车价格预测 - v6 Iterative Pseudo Labeling (LB 442.99)
目标: 使用 v5 (LB 442.99) 的预测结果作为更高质量的伪标签，进行第二轮迭代训练。

策略:
1. 加载 train.csv 和 testB.csv。
2. 加载 predictions_pseudo_v5.csv (LB 442.99)。
3. 特征工程 (复用 train_optimized.py)。
4. 训练模型 (LGB, XGB, CatBoost)。
   - 参数微调: 适当增加模型复杂度或减少正则化，以利用更多“真实”的标签信息。
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
    from train_optimized import build_all_features, LGB_PARAMS, XGB_PARAMS
except ImportError:
    pass

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_FOLDS = 5

# v6 Params: Slightly modified from v2/v5 to capture more nuances
# LightGBM: Increase leaves slightly? 127 is already high. Let's keep it.
# Maybe reduce regularization slightly as we trust labels more.
LGB_PARAMS_V6 = LGB_PARAMS.copy()
LGB_PARAMS_V6['reg_alpha'] = 0.5  # Reduced from 1.0 (v2)
LGB_PARAMS_V6['reg_lambda'] = 0.5 # Reduced from 1.0 (v2)

XGB_PARAMS_V6 = XGB_PARAMS.copy()
# XGB v2 used reg_alpha=0.5. Let's keep it.

def train_lgb_v6(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """LightGBM v6 Iteration"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"LightGBM (v6) {n_folds}-Fold Iteration")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        # Concat with Pseudo Labels (Important: Use pd.Series for y_test_log)
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            LGB_PARAMS_V6, dtrain, 
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
        test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        print(f"  Fold {fold+1} MAE: {mean_absolute_error(y.iloc[val_idx], val_pred):.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ LightGBM (v6) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_xgb_v6(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """XGBoost v6 Iteration"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"XGBoost (v6) {n_folds}-Fold Iteration")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            XGB_PARAMS_V6, dtrain,
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
    print(f"\n  ★ XGBoost (v6) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_catboost_v6(X, y, X_test, y_test_pseudo, cat_features, n_folds=N_FOLDS):
    """CatBoost v6 Iteration"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"CatBoost (v6) {n_folds}-Fold Iteration")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.03, # Keep steady
            depth=8,
            l2_leaf_reg=3, # Reduced from 5 (v2), trusting data more
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
    print(f"\n  ★ CatBoost (v6) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def main():
    print("=" * 60)
    print("二手车价格预测 - v6 Iterative Pseudo Labeling (Load v5 Best)")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/6] 加载数据 & v5 伪标签...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    
    # Load v5 Predictions (LB 442.99)
    pseudo_path = "../prediction_result/predictions_pseudo_v5.csv"
    if not os.path.exists(pseudo_path):
        print(f"Error: v5 Pseudo label file {pseudo_path} not found!")
        return
        
    pseudo_df = pd.read_csv(pseudo_path)
    # Align check
    if not np.array_equal(pseudo_df["SaleID"].values, test["SaleID"].values):
        print("Aligning pseudo labels...")
        pseudo_df = pd.DataFrame({"SaleID": test["SaleID"]}).merge(pseudo_df, on="SaleID", how="left")
        
    y_test_pseudo = pseudo_df["price"].values
    
    # 2. Feat Eng
    print("\n[2/6] 特征工程...")
    try:
        from train_optimized import build_all_features
    except ImportError:
        pass
    train, test = build_all_features(train, test)
    
    # 3. Prep Data
    print("\n[3/6] 准备特征...")
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
    
    # 4. Training
    print("\n[4/6] 模型训练 (v6 Iteration)...")
    
    lgb_oof, lgb_test, lgb_mae = train_lgb_v6(X, y, X_test, y_test_pseudo, feature_cols)
    xgb_oof, xgb_test, xgb_mae = train_xgb_v6(X, y, X_test, y_test_pseudo, feature_cols)
    
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols_for_cb:
        X_cb[c] = X_cb[c].astype(int)
        X_test_cb[c] = X_test_cb[c].astype(int)
    cb_oof, cb_test, cb_mae = train_catboost_v6(X_cb, y, X_test_cb, y_test_pseudo, cat_features=cat_cols_for_cb)
    
    # 5. Ensemble
    print("\n[5/6] 再次融合 (Weighted)...")
    final_pred = 0.4 * lgb_test + 0.2 * xgb_test + 0.4 * cb_test
    
    # 6. Save
    print("\n[6/6] 保存 v6 结果...")
    os.makedirs("../prediction_result", exist_ok=True)
    submission = pd.DataFrame({"SaleID": test["SaleID"], "price": final_pred})
    submission["price"] = submission["price"].clip(lower=0)
    
    out_path = "../prediction_result/predictions_pseudo_v6.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
    print(f"LGB v6 MAE: {lgb_mae:.2f}")
    print(f"XGB v6 MAE: {xgb_mae:.2f}")
    print(f"Cat v6 MAE: {cb_mae:.2f}")

if __name__ == "__main__":
    main()
