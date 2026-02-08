"""
主程序入口
使用方法: python main.py
执行完成后，预测结果将保存在 ../prediction_result/predictions.csv
"""
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# 导入自定义模块
from feature.feature_engineering import build_features
from model.train_model import train_lgb_kfold


def main():
    print("=" * 60)
    print("天池二手车价格预测 - 比赛提交版本")
    print("=" * 60)
    
    # ============== 1. 加载数据 ==============
    print("\n[1/5] 加载数据...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testA_20200313.csv", sep=" ")
    print(f"训练集: {train.shape}, 测试集: {test.shape}")
    
    # 保存测试集 SaleID
    test_ids = test["SaleID"].values
    
    # ============== 2. 特征工程 ==============
    print("\n[2/5] 特征工程...")
    train = build_features(train)
    test = build_features(test)
    print("特征工程完成")
    
    # ============== 3. 准备特征 ==============
    print("\n[3/5] 准备特征...")
    
    # 删除不需要的列
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType"]
    feature_cols = [col for col in train.columns if col not in drop_cols + ["price"]]
    
    # 标签编码
    for col in feature_cols:
        if train[col].dtype == "object":
            le = LabelEncoder()
            train[col] = train[col].fillna("unknown")
            test[col] = test[col].fillna("unknown")
            combined = pd.concat([train[col], test[col]])
            le.fit(combined)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
    
    # 填充缺失值
    for col in feature_cols:
        if train[col].isnull().any():
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)
    
    X = train[feature_cols]
    y = train["price"]
    X_test = test[feature_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    
    # ============== 4. 模型训练 ==============
    print("\n[4/5] LightGBM 5-Fold 训练...")
    
    # 切换工作目录到 code/
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    oof_preds, test_preds, models = train_lgb_kfold(
        X, y, X_test,
        use_log=True,
        save_model=True
    )
    
    os.chdir(original_dir)
    
    # ============== 5. 保存预测结果 ==============
    print("\n[5/5] 保存预测结果...")
    
    # 确保输出目录存在
    os.makedirs("../prediction_result", exist_ok=True)
    
    # 生成提交文件
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": test_preds
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    output_path = "../prediction_result/predictions.csv"
    submission.to_csv(output_path, index=False)
    print(f"预测结果已保存: {output_path}")
    
    # 显示预测结果统计
    print(f"\n预测结果统计:")
    print(f"  样本数: {len(submission)}")
    print(f"  价格均值: {submission['price'].mean():.2f}")
    print(f"  价格范围: {submission['price'].min():.2f} ~ {submission['price'].max():.2f}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
