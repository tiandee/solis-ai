"""
Stacking Fusion Script
Combines OOF predictions from LightGBM, XGBoost, CatBoost, and Neural Network using a Meta-Learner.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge, RidgeCV, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import os
import warnings

warnings.filterwarnings("ignore")

def load_oof_and_preds():
    """Load OOF and Test predictions from CSV files"""
    print("Loading OOF and Test predictions...")
    
    # 1. Load True Target
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    
    # Base path for predictions
    pred_dir = "../prediction_result"
    
    models = ["lgb", "xgb", "cat", "nn"]
    oof_dfs = []
    test_dfs = []
    
    for model in models:
        oof_path = os.path.join(pred_dir, f"oof_{model}.csv")
        test_path = os.path.join(pred_dir, f"predictions_{model}.csv") if model == "nn" else os.path.join(pred_dir, f"pred_{model}.csv")
        
        if not os.path.exists(oof_path):
            print(f"Warning: {oof_path} not found. Skipping {model}.")
            continue
            
        # Read OOF
        oof = pd.read_csv(oof_path)
        oof = oof.rename(columns={"price": f"pred_{model}"})
        oof_dfs.append(oof)
        
        # Read Test Preds
        if os.path.exists(test_path):
            t_pred = pd.read_csv(test_path)
            t_pred = t_pred.rename(columns={"price": f"pred_{model}"})
            test_dfs.append(t_pred)
        else:
            print(f"Warning: {test_path} not found.")

    if not oof_dfs:
        raise ValueError("No OOF predictions found!")

    # Merge OOFs
    meta_train = train[["SaleID", "price"]].copy()
    for oof in oof_dfs:
        meta_train = meta_train.merge(oof, on="SaleID", how="left")
        
    # Merge Test Preds
    meta_test = test[["SaleID"]].copy()
    for t_pred in test_dfs:
        meta_test = meta_test.merge(t_pred, on="SaleID", how="left")
        
    return meta_train, meta_test

def train_meta_model(meta_train, meta_test):
    """Train Meta Model (Stacking)"""
    print("\nTraining Meta Model...")
    
    feature_cols = [c for c in meta_train.columns if c.startswith("pred_")]
    print(f"Meta Features: {feature_cols}")
    
    X = meta_train[feature_cols].values
    y = meta_train["price"].values
    X_test = meta_test[feature_cols].values

    # Log transform input features (predictions) to match target scale
    X = np.log1p(X)
    X_test = np.log1p(X_test)
    
    # Using Log transform for target in Stacking too
    y_log = np.log1p(y)
    
    # Meta Model: BayesianRidge is robust
    meta_model = BayesianRidge()
    # meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    
    # KFold Stacking (Optional, but safer to check CV)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]
        
        meta_model.fit(X_tr, y_tr)
        
        val_pred = np.expm1(meta_model.predict(X_val))
        meta_oof[val_idx] = val_pred
        
        test_preds += np.expm1(meta_model.predict(X_test)) / 5
        
        fold_mae = mean_absolute_error(y[val_idx], val_pred)
        print(f"  Fold {fold+1} Meta MAE: {fold_mae:.2f}")
        
    cv_mae = mean_absolute_error(y, meta_oof)
    print(f"\n★ Stacking CV MAE: {cv_mae:.2f}")
    
    # Check Weights (Coefficients)
    meta_model.fit(X, y_log) # Refit on full data for interpretation
    print("\nMeta Model Coefficients:")
    for feat, coef in zip(feature_cols, meta_model.coef_):
        print(f"  {feat}: {coef:.4f}")
        
    # Final Predict on Test (using the averaged test_preds from CV is better)
    final_pred = test_preds
    
    return final_pred, cv_mae

def main():
    meta_train, meta_test = load_oof_and_preds()
    
    final_pred, cv_mae = train_meta_model(meta_train, meta_test)
    
    # Save Submission
    submission = pd.DataFrame({
        "SaleID": meta_test["SaleID"],
        "price": final_pred
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    output_path = "../prediction_result/predictions_stacking.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"\nStacking Submission Saved: {output_path}")
    print(f"Expected Score: {cv_mae:.2f} (Locally validated)")

if __name__ == "__main__":
    main()
