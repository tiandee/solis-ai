"""
Weighted Blending with Neural Network
Optimizes weights for LightGBM, XGBoost, CatBoost, and NN to minimize MAE.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
import os

def main():
    print("Loading OOF predictions...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test_ids = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")["SaleID"]
    
    y_true = train["price"].values
    
    # Load OOFs
    pred_dir = "../prediction_result"
    
    models = ["lgb", "xgb", "cat", "nn"]
    oofs = []
    tests = []
    
    for model in models:
        oof_path = os.path.join(pred_dir, f"oof_{model}.csv")
        test_path = os.path.join(pred_dir, f"predictions_{model}.csv") if model == "nn" else os.path.join(pred_dir, f"pred_{model}.csv")
        
        if not os.path.exists(oof_path):
            print(f"Warning: {oof_path} not found. Skipping.")
            continue
            
        # OOF needs to be merged on SaleID to ensure order? 
        # train_optimized saved them ordered by train['SaleID'], so simplistic load is fine if consistent.
        # But to be safe, let's merge.
        oof_df = pd.read_csv(oof_path)
        # Ensure it aligns with y_true (which is from train csv)
        # train csv might be shuffled if not read carefully? No, read_csv keeps order.
        # Check alignment:
        if not np.array_equal(oof_df["SaleID"].values, train["SaleID"].values):
             # Align
             oof_df = train[["SaleID"]].merge(oof_df, on="SaleID", how="left")
        
        oofs.append(oof_df["price"].values)
        
        # Test
        t_df = pd.read_csv(test_path)
        # Align test
        if not np.array_equal(t_df["SaleID"].values, test_ids.values):
             t_df = pd.DataFrame({"SaleID": test_ids}).merge(t_df, on="SaleID", how="left")
        
        tests.append(t_df["price"].values)
        
    if not oofs:
        print("No models found!")
        return

    oofs = np.array(oofs).T
    tests = np.array(tests).T
    
    print(f"Ensembling {len(models)} models: {models}")
    
    # Optimization
    def mae_func(weights):
        # Normalize weights
        # weights = np.array(weights)
        # weights /= weights.sum()
        final_pred = np.dot(oofs, weights)
        return mean_absolute_error(y_true, final_pred)
    
    # Constraints: sum=1, 0<=w<=1
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(models)
    
    init_weights = [1.0/len(models)] * len(models)
    
    res = minimize(mae_func, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
    
    best_weights = res.x
    best_mae = res.fun
    
    print("\nOptimal Weights:")
    for model, w in zip(models, best_weights):
        print(f"  {model}: {w:.4f}")
        
    print(f"\n★ Weighted Blend CV MAE: {best_mae:.2f}")
    
    # Predict
    final_test_pred = np.dot(tests, best_weights)
    
    # Save
    submission = pd.DataFrame({
        "SaleID": test_ids,
        "price": final_test_pred
    })
    submission["price"] = submission["price"].clip(lower=0)
    
    out_path = "../prediction_result/predictions_ensemble_mae.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission Saved: {out_path}")

if __name__ == "__main__":
    main()
