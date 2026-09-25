# =============================================================================
# Step 5 — Multimodal Transformer
# =============================================================================
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from config import DEVICE,SEED

class ModalityEncoder(nn.Module):
    def __init__(self,in_dim,d_model,dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model,d_model),
        )
    def forward(self,x):
        return self.net(x)

class MultimodalTransformer(nn.Module):
    def __init__(self,modality_dims,d_model=64,n_heads=4,n_layers=2,dropout=0.1,out_dim=1):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(modality_dims[m],d_model,dropout)
            for m in self.modalities
        })
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.cls_token,std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,
            dim_feedforward=d_model*4,dropout=dropout,
            batch_first=True,activation="gelu")
        self.transformer = nn.TransformerEncoder(layer,num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model,out_dim),
        )
    def _tokenize(self,x,modality_slices):
        tokens = []
        for m in self.modalities:
            idx = modality_slices[m]
            if len(idx) == 0:
                continue
            tokens.append(self.encoders[m](x[:,idx]))
        return torch.stack(tokens,dim=1)
    def forward(self,x,modality_slices):
        tokens = self._tokenize(x,modality_slices)
        B = tokens.size(0)
        cls = self.cls_token.expand(B,-1,-1)
        seq = torch.cat([cls,tokens],dim=1)
        z = self.transformer(seq)
        return self.head(z[:,0]).squeeze(-1)
    def extract_fused(self,x,modality_slices):
        tokens = self._tokenize(x,modality_slices)
        B = tokens.size(0)
        cls = self.cls_token.expand(B,-1,-1)
        seq = torch.cat([cls,tokens],dim=1)
        z = self.transformer(seq)
        return z[:,0]

def compute_metrics(y_true_raw,y_pred_raw):
    rmse = float(np.sqrt(mean_squared_error(y_true_raw,y_pred_raw)))
    return {
        "R2":float(r2_score(y_true_raw,y_pred_raw)),
        "RMSE":rmse,
        "MAE":float(mean_absolute_error(y_true_raw,y_pred_raw)),
        "bias":float(np.mean(y_pred_raw-y_true_raw)),
        "rRMSE":rmse/(float(np.mean(y_true_raw))+1e-8),
    }

def train_transformer(data,d_model=64,n_heads=4,n_layers=2,epochs=300,lr=1e-3,batch_size=16,patience=40,verbose=False):
    print("\n"+"="*70)
    print("STEP 5: Training Multimodal Transformer")
    print("="*70)
    torch.manual_seed(SEED)
    X_tr = torch.tensor(data["X_train"],dtype=torch.float32)
    y_tr = torch.tensor(data["y_train"],dtype=torch.float32)
    X_va = torch.tensor(data["X_val"],dtype=torch.float32).to(DEVICE)
    y_va = torch.tensor(data["y_val"],dtype=torch.float32).to(DEVICE)
    X_te = torch.tensor(data["X_test"],dtype=torch.float32).to(DEVICE)
    modality_dims = {m:len(idx) for m,idx in data["modality_slices"].items() if len(idx)>0}
    model = MultimodalTransformer(modality_dims,d_model,n_heads,n_layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr,y_tr),batch_size=batch_size,shuffle=True)
    best_val,best_state,bad = np.inf,None,0
    t0 = time.time()
    for epoch in range(1,epochs+1):
        model.train()
        for xb,yb in loader:
            xb,yb = xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb,data["modality_slices"])
            loss = loss_fn(pred,yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va,data["modality_slices"]),y_va).item()
        if val_loss < best_val-1e-5:
            best_val,bad = val_loss,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[tm] Early stop at epoch {epoch}")
                break
        if verbose:
            print(f"[tm] Epoch {epoch:3d}/{epochs} | val_loss={val_loss:.4f} | lr={opt.param_groups[0]['lr']:.6f}")
    model.load_state_dict(best_state)
    model.eval()
    train_time = time.time()-t0
    with torch.no_grad():
        Z_train = model.extract_fused(torch.tensor(data["X_train"],dtype=torch.float32).to(DEVICE),data["modality_slices"]).cpu().numpy()
        Z_val = model.extract_fused(torch.tensor(data["X_val"],dtype=torch.float32).to(DEVICE),data["modality_slices"]).cpu().numpy()
        Z_test = model.extract_fused(torch.tensor(data["X_test"],dtype=torch.float32).to(DEVICE),data["modality_slices"]).cpu().numpy()
        y_hat_test_s = model(X_te,data["modality_slices"]).cpu().numpy()
    y_hat_test = y_hat_test_s*data["y_std"]+data["y_mean"]
    metrics = compute_metrics(data["y_test_raw"],y_hat_test)
    metrics["train_time_s"] = round(train_time,2)
    metrics["n_params"] = int(sum(p.numel() for p in model.parameters()))
    return model,{"Z_train":Z_train,"Z_val":Z_val,"Z_test":Z_test,"y_pred_test":y_hat_test},metrics