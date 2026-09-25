# =============================================================================
# MDQ-BioMap Data Loading and Preprocessing
# =============================================================================
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from config import TARGET, ID_COLS, CAT_COL, MODALITIES, SEED

def load_and_split(csv_path, train_frac=0.70, val_frac=0.15, seed=SEED):
    print(f"\n[data] Loading {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[data] Raw shape: {df.shape}")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in CSV.\nAvailable columns: {list(df.columns)}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[data] After dropping NaN target: {df.shape}")
    y = df[TARGET].values.astype(np.float32)
    drop_cols = [TARGET] + [c for c in ID_COLS if c in df.columns]
    X_df = df.drop(columns=drop_cols)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df,y,test_size=val_frac,random_state=seed)
    val_ratio = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,y_trainval,test_size=val_ratio,random_state=seed)
    print(f"[data] Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    numeric_cols = [c for c in X_df.columns if c != CAT_COL]
    cat_cols = [CAT_COL] if CAT_COL in X_df.columns else []
    num_pipe = Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehotencoder",OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = [("num",num_pipe,numeric_cols)]
    if cat_cols:
        transformers.append(("cat",cat_pipe,cat_cols))
    pre = ColumnTransformer(transformers,remainder="drop")
    pre.fit(X_train)
    X_train_p = pre.transform(X_train).astype(np.float32)
    X_val_p = pre.transform(X_val).astype(np.float32)
    X_test_p = pre.transform(X_test).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)
    y_train_s = (y_train-y_mean)/y_std
    y_val_s = (y_val-y_mean)/y_std
    y_test_s = (y_test-y_mean)/y_std
    feat_names = []
    for name,trans,cols in pre.transformers_:
        if name == "num":
            feat_names.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps.get("onehotencoder")
            if ohe is not None:
                feat_names.extend(ohe.get_feature_names_out(cols).tolist())
    if len(feat_names) != X_train_p.shape[1]:
        while len(feat_names) < X_train_p.shape[1]:
            feat_names.append(f"feat_{len(feat_names)}")
        feat_names = feat_names[:X_train_p.shape[1]]
    modality_slices = {}
    for mod,cols in MODALITIES.items():
        idxs = [i for i,f in enumerate(feat_names)
                if f in cols or any(f.startswith(c+"_") for c in cols)]
        modality_slices[mod] = idxs
        print(f"[data] Modality '{mod}': {len(idxs)} features")
    return {
        "X_train":X_train_p,"X_val":X_val_p,"X_test":X_test_p,
        "y_train":y_train_s,"y_val":y_val_s,"y_test":y_test_s,
        "y_train_raw":y_train,"y_val_raw":y_val,"y_test_raw":y_test,
        "y_mean":y_mean,"y_std":y_std,
        "feat_names":feat_names,"modality_slices":modality_slices,
        "preprocessor":pre,"df":df,
        "X_train_df":X_train,"X_val_df":X_val,"X_test_df":X_test,
    }