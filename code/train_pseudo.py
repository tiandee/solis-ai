"""
二手车价格预测 - v5 Pseudo Labeling
目标: 利用 LB 445 的高分预测结果作为伪标签 (Pseudo Labels) 扩充训练集，进一步提升泛化能力。

策略:
1. 加载 train.csv 和 testB.csv (及其预测结果 predictions_ensemble_mae.csv)。
2. 特征工程 (复用 train_optimized.py)。
3. 在 K-Fold 训练时:
    - 训练集 = 原始训练集(Fold_train) + 伪标签测试集(TestB)
    - 验证集 = 原始训练集(Fold_valid)  <- 严禁包含伪标签数据!
    - 目标函数: 回归到 Log-Target (v2 方案，更稳健)
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

def train_lgb_pseudo(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """LightGBM Pseudo Labeling 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # Target Log Transform (v2 Strategy)
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"LightGBM (Pseudo) {n_folds}-Fold 训练")
    print(f"{'='*50}")
    
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X)):
        # 原始训练集切分
        X_tr_orig, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_orig, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
        
        # 注入伪标签数据
        # 训练集 = Fold_train + Full_Pseudo_Test
        X_tr = pd.concat([X_tr_orig, X_test], axis=0)
        y_tr = pd.concat([y_tr_orig, pd.Series(y_test_log)], axis=0)
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            LGB_PARAMS, dtrain, # Reuse v2 Params
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
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ LightGBM (Pseudo) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_xgb_pseudo(X, y, X_test, y_test_pseudo, feature_cols, n_folds=N_FOLDS):
    """XGBoost Pseudo Labeling 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"XGBoost (Pseudo) {n_folds}-Fold 训练")
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
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ XGBoost (Pseudo) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def train_catboost_pseudo(X, y, X_test, y_test_pseudo, cat_features, n_folds=N_FOLDS):
    """CatBoost Pseudo Labeling 训练"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    y_log = np.log1p(y)
    y_test_log = np.log1p(y_test_pseudo)
    
    print(f"\n{'='*50}")
    print(f"CatBoost (Pseudo) {n_folds}-Fold 训练")
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
            random_seed=RANDOM_SEED,
            verbose=500,
            early_stopping_rounds=200,
            eval_metric="MAE", # Metric
            loss_function='MAE', # Loss: Use MAE or RMSE? v2 used default (RMSE usually). 
            # Wait, v2 used RMSE (Log-Target). Let's stick to RMSE for stability, but eval MAE.
            # CatBoost default is RMSE.
            cat_features=cat_features,
        )
        
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        
        val_pred = np.expm1(model.predict(X_val)).clip(0)
        oof[val_idx] = val_pred
        
        test_preds += np.expm1(model.predict(X_test)).clip(0) / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[val_idx], val_pred)
        print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")
    
    cv_mae = mean_absolute_error(y, oof)
    print(f"\n  ★ CatBoost (Pseudo) CV MAE: {cv_mae:.2f}")
    return oof, test_preds, cv_mae


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("二手车价格预测 - v5 Pseudo Labeling (LB 445)")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据 & 伪标签...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    
    # 加载 Pseudo Labels
    pseudo_path = "../prediction_result/predictions_ensemble_mae.csv"
    if not os.path.exists(pseudo_path):
        print(f"Error: Pseudo label file {pseudo_path} not found!")
        return
        
    pseudo_df = pd.read_csv(pseudo_path)
    # Align
    if not np.array_equal(pseudo_df["SaleID"].values, test["SaleID"].values):
        print("Aligning pseudo labels...")
        pseudo_df = pd.DataFrame({"SaleID": test["SaleID"]}).merge(pseudo_df, on="SaleID", how="left")
        
    y_test_pseudo = pseudo_df["price"].values
    test["price"] = -1 # Placeholder
    
    print(f"训练集: {train.shape}, 测试集: {test.shape} (with Pseudo Labels)")
    
    test_ids = test["SaleID"].values
    
    # 2. 特征工程 (复用)
    print("\n[2/6] 特征工程...")
    try:
        from train_optimized import build_all_features
    except ImportError:
        print("Error: train_optimized.py not found.")
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
            
    # Label Encode
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
    print("\n[4/6] 模型训练 (Pseudo Labeling)...")
    
    # Train ALL models (LGB, XGB, Cat)
    lgb_oof, lgb_test, lgb_mae = train_lgb_pseudo(X, y, X_test, y_test_pseudo, feature_cols)
    xgb_oof, xgb_test, xgb_mae = train_xgb_pseudo(X, y, X_test, y_test_pseudo, feature_cols)
    
    # CatBoost Prep
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols_for_cb:
        X_cb[c] = X_cb[c].astype(int)
        X_test_cb[c] = X_test_cb[c].astype(int)
        
    cb_oof, cb_test, cb_mae = train_catboost_pseudo(X_cb, y, X_test_cb, y_test_pseudo, cat_features=cat_cols_for_cb)
    
    # 5. 再次融合 (Weighted)
    print("\n[5/6] 再次融合 (Weighted Blending)...")
    
    # 简单的加权平均，或者复用 minimize 找权重
    # 这里直接用平均，或者 0.4, 0.2, 0.4
    final_pred = 0.4 * lgb_test + 0.2 * xgb_test + 0.4 * cb_test
    
    # 6. 保存
    print("\n[6/6] 保存结果...")
    os.makedirs("../prediction_result", exist_ok=True)
    
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": final_pred
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    out_path = "../prediction_result/predictions_pseudo_v5.csv"
    submission.to_csv(out_path, index=False)
    
    print(f"\nSaved to: {out_path}")
    print(f"LGB Pseudo MAE: {lgb_mae:.2f}")
    print(f"XGB Pseudo MAE: {xgb_mae:.2f}")
    print(f"Cat Pseudo MAE: {cb_mae:.2f}")

if __name__ == "__main__":
    main()
