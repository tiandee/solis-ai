"""
项目配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 模型路径
MODELS_DIR = PROJECT_ROOT / "models"

# 提交路径
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 数据文件名
TRAIN_FILE = "used_car_train_20200313.csv"
TEST_A_FILE = "used_car_testA_20200313.csv"
TEST_B_FILE = "used_car_testB_20200421.csv"

# 随机种子
RANDOM_SEED = 42

# 交叉验证折数
N_FOLDS = 5

# 特征配置
NUMERIC_FEATURES = [
    'power', 'kilometer', 
    'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7',
    'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14'
]

CATEGORICAL_FEATURES = [
    'name', 'model', 'brand', 'bodyType', 'fuelType', 
    'gearbox', 'notRepairedDamage', 'regionCode', 'seller', 'offerType'
]

DATE_FEATURES = ['regDate', 'creatDate']

TARGET = 'price'

# LightGBM 默认参数
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': -1
}

# XGBoost 默认参数
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbosity': 0
}
