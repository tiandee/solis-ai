"""
模型训练模块
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from typing import Tuple, Dict, Any
import warnings

from .config import LGB_PARAMS, XGB_PARAMS, RANDOM_SEED, N_FOLDS

warnings.filterwarnings('ignore')


def train_lgb_kfold(X: pd.DataFrame, y: pd.Series, 
                    X_test: pd.DataFrame = None,
                    params: Dict = None,
                    n_folds: int = N_FOLDS,
                    use_log: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    LightGBM K-Fold 交叉验证训练
    
    Args:
        X: 训练特征
        y: 目标变量
        X_test: 测试特征
        params: 模型参数
        n_folds: 折数
        use_log: 是否对目标做对数变换
    
    Returns:
        oof_preds: 验证集预测
        test_preds: 测试集预测
        models: 训练的模型列表
    """
    if params is None:
        params = LGB_PARAMS.copy()
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    
    # 对数变换
    if use_log:
        y_train = np.log1p(y)
    else:
        y_train = y
    
    print(f"开始 LightGBM {n_folds}-Fold 训练...")
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        train_data = lgb.Dataset(X_train, label=y_tr)
        valid_data = lgb.Dataset(X_valid, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=10000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=200)
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
    
    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\n{'='*50}")
    print(f"Overall CV MAE: {overall_mae:.4f}")
    print(f"{'='*50}")
    
    return oof_preds, test_preds, models


def train_xgb_kfold(X: pd.DataFrame, y: pd.Series,
                    X_test: pd.DataFrame = None,
                    params: Dict = None,
                    n_folds: int = N_FOLDS,
                    use_log: bool = True) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    XGBoost K-Fold 交叉验证训练
    """
    if params is None:
        params = XGB_PARAMS.copy()
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    models = []
    
    if use_log:
        y_train = np.log1p(y)
    else:
        y_train = y
    
    print(f"开始 XGBoost {n_folds}-Fold 训练...")
    
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        dtrain = xgb.DMatrix(X_train, label=y_tr)
        dvalid = xgb.DMatrix(X_valid, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=200
        )
        
        models.append(model)
        
        val_pred = model.predict(dvalid)
        if use_log:
            val_pred = np.expm1(val_pred)
            val_pred = np.clip(val_pred, 0, None)
        
        oof_preds[valid_idx] = val_pred
        
        if X_test is not None:
            dtest = xgb.DMatrix(X_test)
            test_pred = model.predict(dtest)
            if use_log:
                test_pred = np.expm1(test_pred)
                test_pred = np.clip(test_pred, 0, None)
            test_preds += test_pred / n_folds
        
        fold_mae = mean_absolute_error(y.iloc[valid_idx], val_pred)
        print(f"Fold {fold + 1} MAE: {fold_mae:.4f}")
    
    overall_mae = mean_absolute_error(y, oof_preds)
    print(f"\n{'='*50}")
    print(f"Overall CV MAE: {overall_mae:.4f}")
    print(f"{'='*50}")
    
    return oof_preds, test_preds, models


def blend_predictions(predictions_list: list, weights: list = None) -> np.ndarray:
    """
    模型融合 - 加权平均
    
    Args:
        predictions_list: 预测结果列表
        weights: 权重列表（默认等权重）
    
    Returns:
        融合后的预测结果
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    
    assert len(predictions_list) == len(weights), "预测数量与权重数量不匹配"
    assert abs(sum(weights) - 1.0) < 1e-6, "权重之和必须为1"
    
    blended = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        blended += pred * weight
    
    return blended
