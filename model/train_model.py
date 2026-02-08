"""
模型训练模块
按照比赛规范放置在 model/ 目录
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import joblib
import os


RANDOM_SEED = 42
N_FOLDS = 5

LGB_PARAMS = {
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


def train_lgb_kfold(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame = None,
                    params: dict = None, n_folds: int = N_FOLDS, 
                    use_log: bool = True, save_model: bool = True):
    """
    LightGBM K-Fold 交叉验证训练
    
    Args:
        X: 训练特征
        y: 目标变量
        X_test: 测试特征
        params: LightGBM 参数
        n_folds: 折数
        use_log: 是否对目标变量进行 log1p 变换
        save_model: 是否保存模型
    
    Returns:
        oof_preds: OOF 预测
        test_preds: 测试集预测 (如果提供了 X_test)
        models: 训练好的模型列表
    """
    if params is None:
        params = LGB_PARAMS
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    
    # 对数变换
    y_train_full = np.log1p(y) if use_log else y
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*20} Fold {fold + 1}/{n_folds} {'='*20}")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=500)
            ]
        )
        
        models.append(model)
        
        # 验证集预测
        val_pred = model.predict(X_valid)
        if use_log:
            val_pred = np.expm1(val_pred)
        val_pred = np.clip(val_pred, 0, None)
        oof_preds[valid_idx] = val_pred
        
        # 测试集预测
        if X_test is not None:
            test_pred = model.predict(X_test)
            if use_log:
                test_pred = np.expm1(test_pred)
            test_pred = np.clip(test_pred, 0, None)
            test_preds += test_pred / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[valid_idx], val_pred)
        print(f"Fold {fold + 1} MAE: {fold_mae:.4f}")
        
        # 保存模型
        if save_model:
            os.makedirs("../model", exist_ok=True)
            joblib.dump(model, f"../model/lgb_fold{fold + 1}.pkl")
    
    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\n{'='*50}")
    print(f"Overall CV MAE: {overall_mae:.4f}")
    
    return oof_preds, test_preds, models


def load_models(model_dir: str = "../model", n_folds: int = N_FOLDS):
    """加载已保存的模型"""
    models = []
    for fold in range(1, n_folds + 1):
        model_path = os.path.join(model_dir, f"lgb_fold{fold}.pkl")
        if os.path.exists(model_path):
            models.append(joblib.load(model_path))
    return models


def predict_with_models(models: list, X_test: pd.DataFrame, use_log: bool = True):
    """使用多个模型进行预测并平均"""
    test_preds = np.zeros(len(X_test))
    
    for model in models:
        pred = model.predict(X_test)
        if use_log:
            pred = np.expm1(pred)
        pred = np.clip(pred, 0, None)
        test_preds += pred / len(models)
    
    return test_preds
