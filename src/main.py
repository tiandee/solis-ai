"""
主程序入口
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import (
    RAW_DATA_DIR, TRAIN_FILE, TEST_A_FILE, 
    SUBMISSIONS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
)
from src.utils import load_data, reduce_memory_usage, save_submission
from src.features import build_features
from src.models import train_lgb_kfold, train_xgb_kfold, blend_predictions


def main():
    """主函数"""
    print("=" * 60)
    print("天池二手车价格预测 - 模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_path = RAW_DATA_DIR / TRAIN_FILE
    test_path = RAW_DATA_DIR / TEST_A_FILE
    
    if not train_path.exists():
        print(f"错误: 找不到训练数据文件 {train_path}")
        print("请先从天池下载数据并放入 data/raw/ 目录")
        return
    
    train_df = load_data(train_path)
    test_df = load_data(test_path) if test_path.exists() else None
    
    print(f"训练集大小: {train_df.shape}")
    if test_df is not None:
        print(f"测试集大小: {test_df.shape}")
    
    # 2. 特征工程
    print("\n[2/5] 特征工程...")
    train_processed, test_processed = build_features(train_df, test_df)
    
    # 3. 准备训练数据
    print("\n[3/5] 准备训练数据...")
    
    # 删除不需要的列
    drop_cols = ['SaleID', 'name', 'regDate', 'creatDate', 'seller', 'offerType']
    feature_cols = [col for col in train_processed.columns 
                    if col not in drop_cols + [TARGET]]
    
    # 标签编码
    le_dict = {}
    for col in feature_cols:
        if train_processed[col].dtype == 'object':
            le = LabelEncoder()
            train_processed[col] = le.fit_transform(train_processed[col].astype(str))
            if test_processed is not None:
                test_processed[col] = le.transform(test_processed[col].astype(str))
            le_dict[col] = le
    
    X = train_processed[feature_cols]
    y = train_processed[TARGET]
    X_test = test_processed[feature_cols] if test_processed is not None else None
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 4. 模型训练
    print("\n[4/5] 模型训练...")
    
    # LightGBM
    lgb_oof, lgb_test, lgb_models = train_lgb_kfold(X, y, X_test)
    
    # XGBoost
    xgb_oof, xgb_test, xgb_models = train_xgb_kfold(X, y, X_test)
    
    # 模型融合
    if X_test is not None:
        print("\n[5/5] 模型融合与提交...")
        
        # 加权融合 (LGB:XGB = 0.6:0.4)
        final_pred = blend_predictions([lgb_test, xgb_test], weights=[0.6, 0.4])
        
        # 保存提交文件
        save_submission(final_pred, test_df, "submit.csv", SUBMISSIONS_DIR)
        
        print("\n" + "=" * 60)
        print("完成！提交文件已生成。")
        print("=" * 60)
    else:
        print("\n未提供测试集，跳过提交文件生成。")


if __name__ == "__main__":
    main()
