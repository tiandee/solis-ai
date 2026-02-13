"""
使用已训练的模型对 testB 生成预测结果
无需重新训练，直接加载保存的模型
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

from feature.feature_engineering import build_features
from model.train_model import load_models, predict_with_models


def main():
    print("=" * 60)
    print("使用已有模型预测 testB")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    print(f"训练集: {train.shape}, 测试集B: {test.shape}")
    print(f"测试集 SaleID 范围: {test['SaleID'].min()} ~ {test['SaleID'].max()}")

    test_ids = test["SaleID"].values

    # 2. 特征工程
    print("\n[2/4] 特征工程...")
    train = build_features(train)
    test = build_features(test)

    # 3. 准备特征（需要和训练时一致）
    print("\n[3/4] 准备特征...")
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType"]
    feature_cols = [col for col in train.columns if col not in drop_cols + ["price"]]

    # 标签编码（需要在 train+test 上 fit，保持和训练时一致）
    for col in feature_cols:
        if train[col].dtype == "object":
            le = LabelEncoder()
            train[col] = train[col].fillna("unknown")
            test[col] = test[col].fillna("unknown")
            combined = pd.concat([train[col], test[col]])
            le.fit(combined)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

    for col in feature_cols:
        if train[col].isnull().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
        if test[col].isnull().any():
            median_val = train[col].median()
            test[col] = test[col].fillna(median_val)

    X_test = test[feature_cols]
    print(f"特征数量: {len(feature_cols)}")

    # 4. 加载模型并预测
    print("\n[4/4] 加载模型并预测...")
    models = load_models("../model")
    print(f"加载了 {len(models)} 个模型")

    test_preds = predict_with_models(models, X_test, use_log=True)

    # 保存结果
    os.makedirs("../prediction_result", exist_ok=True)
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": test_preds
    })
    submission["price"] = submission["price"].clip(lower=0)

    output_path = "../prediction_result/predictions_testB.csv"
    submission.to_csv(output_path, index=False)

    print(f"\n预测结果已保存: {output_path}")
    print(f"样本数: {len(submission)}")
    print(f"价格均值: {submission['price'].mean():.2f}")
    print(f"价格范围: {submission['price'].min():.2f} ~ {submission['price'].max():.2f}")

    # 显示前几行
    print(f"\n前5行预览:")
    print(submission.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("完成！请将 prediction_result/predictions_testB.csv 提交到天池")
    print("=" * 60)


if __name__ == "__main__":
    main()
