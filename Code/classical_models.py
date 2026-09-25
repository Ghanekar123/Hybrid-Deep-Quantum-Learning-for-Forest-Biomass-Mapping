# =============================================================================
# Step 7 — Classical Machine Learning and Deep Learning Baselines
# =============================================================================
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from config import DEVICE,SEED
from transformer import compute_metrics

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] XGBoost not installed. XGBoost baseline will be skipped.")

class TorchMLP(nn.Module):
    def __init__(self,in_dim,hidden=128,out_dim=1,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden,hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2,out_dim),
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)

class TorchCNN_LSTM(nn.Module):
    def __init__(self,in_dim,hidden=64,out_dim=1):
        super().__init__()
        self.proj = nn.Linear(in_dim,hidden)
        self.lstm = nn.LSTM(hidden,hidden,batch_first=True)
        self.head = nn.Linear(hidden,out_dim)
    def forward(self,x):
        h = self.proj(x).unsqueeze(1)
        o,_ = self.lstm(h)
        return self.head(o[:,-1]).squeeze(-1)

def train_torch_baseline(model,data,epochs=150,lr=1e-3,batch_size=16,patience=25):
    X_tr = torch.tensor(data["X_train"],dtype=torch.float32)
    y_tr = torch.tensor(data["y_train"],dtype=torch.float32)
    X_va = torch.tensor(data["X_val"],dtype=torch.float32).to(DEVICE)
    y_va = torch.tensor(data["y_val"],dtype=torch.float32).to(DEVICE)
    X_te = torch.tensor(data["X_test"],dtype=torch.float32).to(DEVICE)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr,y_tr),batch_size=batch_size,shuffle=True)
    best_val,best_state,bad = np.inf,None,0
    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb),yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_va),y_va).item()
        if vl < best_val-1e-5:
            best_val,bad = vl,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te).cpu().numpy()
    return pred_s*data["y_std"]+data["y_mean"]

def run_baselines(data):
    print("\n"+"="*70)
    print("STEP 7: Classical Baselines")
    print("="*70)
    Xtr,ytr = data["X_train"],data["y_train_raw"]
    Xte,yte = data["X_test"],data["y_test_raw"]
    results,models = {},{}
    print("\n[baseline] RFE-SVM...")
    t0 = time.time()
    svm = SVR(kernel="rbf",C=10,epsilon=0.1)
    n_feat = min(20,Xtr.shape[1])
    rfe = RFE(svm,n_features_to_select=n_feat)
    rfe.fit(Xtr,ytr)
    svm.fit(Xtr[:,rfe.support_],ytr)
    m = compute_metrics(yte,svm.predict(Xte[:,rfe.support_]))
    m["train_time_s"] = round(time.time()-t0,2)
    results["RFE-SVM"],models["RFE-SVM"] = m,(rfe,svm)
    print("\n[baseline] Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=500,random_state=SEED,n_jobs=-1)
    rf.fit(Xtr,ytr)
    m = compute_metrics(yte,rf.predict(Xte))
    m["train_time_s"] = round(time.time()-t0,2)
    results["RandomForest"],models["RandomForest"] = m,rf
    if HAS_XGB:
        print("\n[baseline] XGBoost...")
        t0 = time.time()
        xgb = XGBRegressor(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,random_state=SEED,n_jobs=-1)
        xgb.fit(Xtr,ytr)
        m = compute_metrics(yte,xgb.predict(Xte))
        m["train_time_s"] = round(time.time()-t0,2)
        results["XGBoost"],models["XGBoost"] = m,xgb
    print("\n[baseline] CNN (MLP surrogate)...")
    t0 = time.time()
    cnn_pred = train_torch_baseline(TorchMLP(data["X_train"].shape[1]),data)
    m = compute_metrics(yte,cnn_pred)
    m["train_time_s"] = round(time.time()-t0,2)
    results["CNN"] = m
    print("\n[baseline] CNN-LSTM...")
    t0 = time.time()
    lstm_pred = train_torch_baseline(TorchCNN_LSTM(data["X_train"].shape[1]),data)
    m = compute_metrics(yte,lstm_pred)
    m["train_time_s"] = round(time.time()-t0,2)
    results["CNN-LSTM"] = m
    return results,models