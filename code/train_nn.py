"""
Neural Network Training Script for Stacking
Target: MAE < 450 (Single Model)
"""
import sys, os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings

# Import feature engineering from the optimized script
try:
    from train_optimized import build_all_features
except ImportError:
    # If running directly, add current dir to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_optimized import build_all_features

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
N_FOLDS = 5
BATCH_SIZE = 2048
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class CarPriceNN(nn.Module):
    def __init__(self, num_numeric, emb_dims, hidden_units=[256, 128, 64]):
        super().__init__()
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, dim) for num_cats, dim in emb_dims
        ])
        
        total_emb_dim = sum(dim for _, dim in emb_dims)
        input_dim = num_numeric + total_emb_dim
        
        # Dense Layers
        layers = []
        in_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = units
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x_num, x_cat):
        emb_outs = []
        for i, emb in enumerate(self.embeddings):
            emb_outs.append(emb(x_cat[:, i]))
        
        x_emb = torch.cat(emb_outs, dim=1)
        x = torch.cat([x_num, x_emb], dim=1)
        return self.network(x)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(x_num, x_cat).squeeze()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            output = model(x_num, x_cat).squeeze()
            loss = criterion(output, y)
            total_loss += loss.item() * y.size(0)
            preds.append(output.cpu().numpy())
            
    return total_loss / len(loader.dataset), np.concatenate(preds)

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x_num, x_cat in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            output = model(x_num, x_cat).squeeze()
            preds.append(output.cpu().numpy())
    return np.concatenate(preds)

def main():
    print("加载数据...")
    train = pd.read_csv("../data/used_car_train_20200313.csv", sep=" ")
    test = pd.read_csv("../data/used_car_testB_20200421.csv", sep=" ")
    
    # 2. 特征工程 (复用优化版脚本)
    print("构建特征...")
    train, test = build_all_features(train, test)
    
    # 3. 神经网络预处理
    print("神经网络预处理...")
    drop_cols = ["SaleID", "name", "regDate", "creatDate", "seller", "offerType",
                 "brand_body", "brand_fuel", "price"]
    
    # 分离数值和类别特征
    cat_cols = ["brand", "model", "bodyType", "fuelType", "gearbox", "regionCode", 
                "power_bin", "kilometer_bin"]
    # 确保 cat_cols 在 columns 中
    cat_cols = [c for c in cat_cols if c in train.columns]
    
    num_cols = [c for c in train.columns if c not in drop_cols + cat_cols]
    
    # 填充缺失值
    for col in num_cols:
        train[col] = train[col].fillna(train[col].mean())
        test[col] = test[col].fillna(train[col].mean())
    
    for col in cat_cols:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)
    
    # 标准化数值特征
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train[num_cols])
    test_num = scaler.transform(test[num_cols])
    
    # 编码类别特征
    train_cat = []
    test_cat = []
    emb_dims = []
    
    for col in cat_cols:
        le = LabelEncoder()
        # 将 train 和 test 组合起来编码，处理未知类别
        full_vals = pd.concat([train[col], test[col]]).astype(str)
        le.fit(full_vals)
        
        train_cat.append(le.transform(train[col].astype(str)))
        test_cat.append(le.transform(test[col].astype(str)))
        
        num_classes = len(le.classes_)
        emb_dim = min(50, (num_classes + 1) // 2)
        emb_dims.append((num_classes, emb_dim))
    
    train_cat = np.stack(train_cat, axis=1)
    test_cat = np.stack(test_cat, axis=1)
    
    X_num = torch.tensor(train_num, dtype=torch.float32)
    X_cat = torch.tensor(train_cat, dtype=torch.long)
    y = torch.tensor(np.log1p(train["price"].values), dtype=torch.float32)
    
    X_test_num = torch.tensor(test_num, dtype=torch.float32)
    X_test_cat = torch.tensor(test_cat, dtype=torch.long)
    
    # DataLoader for Test
    test_dataset = TensorDataset(X_test_num, X_test_cat)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. K-Fold 训练
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_total = np.zeros(len(test))
    
    print(f"\n开始 {N_FOLDS}-Fold NN 训练...")
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_num)):
        print(f"Fold {fold+1}/{N_FOLDS}")
        
        X_tr_num, X_val_num = X_num[tr_idx], X_num[val_idx]
        X_tr_cat, X_val_cat = X_cat[tr_idx], X_cat[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        train_dataset = TensorDataset(X_tr_num, X_tr_cat, y_tr)
        val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = CarPriceNN(len(num_cols), emb_dims).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.L1Loss() # MAE Loss (on log target)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_loss = float('inf')
        patience = 7
        early_stop_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_preds_log = validate(model, val_loader, criterion)
            
            # 反变换回原始空间计算真实 MAE
            val_mae_real = mean_absolute_error(np.expm1(y_val.cpu().numpy()), np.expm1(val_preds_log))
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                early_stop_counter = 0
                # print(f"  Epoch {epoch+1}: Val Loss {val_loss:.5f} | Val MAE {val_mae_real:.2f} *")
            else:
                early_stop_counter += 1
                # print(f"  Epoch {epoch+1}: Val Loss {val_loss:.5f} | Val MAE {val_mae_real:.2f}")
                
            if early_stop_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final Val Preds
        _, val_preds_log = validate(model, val_loader, criterion)
        oof_preds[val_idx] = np.expm1(val_preds_log)
        
        # Test Preds
        test_preds_log = predict(model, test_loader)
        test_preds_total += np.expm1(test_preds_log) / N_FOLDS
        
        fold_mae = mean_absolute_error(np.expm1(y_val.cpu().numpy()), oof_preds[val_idx])
        print(f"  Best Val MAE: {fold_mae:.2f}")
    
    # CV Score
    cv_mae = mean_absolute_error(train["price"], oof_preds)
    print(f"\nNN CV MAE: {cv_mae:.2f}")
    
    # Save Results
    os.makedirs("../prediction_result", exist_ok=True)
    
    # Save OOF
    df_oof = pd.DataFrame({"SaleID": train["SaleID"], "price": oof_preds})
    df_oof.to_csv("../prediction_result/oof_nn.csv", index=False)
    
    # Save Test Preds
    df_test = pd.DataFrame({"SaleID": test["SaleID"], "price": test_preds_total})
    df_test.to_csv("../prediction_result/predictions_nn.csv", index=False)
    print("NN OOF and Predictions Saved.")

if __name__ == "__main__":
    main()
