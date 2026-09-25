# =============================================================================
# Step 8 — Cross-Domain Transfer Evaluation
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformer import compute_metrics

def cross_domain_evaluation(df,domain_col,feature_cols,model_factory,target="agb_mg_ha"):
    print("\n"+"="*70)
    print(f"STEP 8: Cross-Domain Evaluation (leave-one-{domain_col}-out)")
    print("="*70)
    if domain_col not in df.columns:
        print(f"[skip] Column '{domain_col}' not in dataframe.")
        return {}
    domains = df[domain_col].dropna().unique()
    print(f"[transfer] Domains: {list(domains)}")
    results = {}
    for held in domains:
        train_df = df[df[domain_col]!=held].dropna(subset=[target])
        test_df = df[df[domain_col]==held].dropna(subset=[target])
        if len(test_df)<3 or len(train_df)<10:
            print(f"[transfer] Skip '{held}' (train={len(train_df)}, test={len(test_df)})")
            continue
        Xtr = train_df[feature_cols].apply(pd.to_numeric,errors="coerce")
        Xte = test_df[feature_cols].apply(pd.to_numeric,errors="coerce")
        med = Xtr.median()
        Xtr = Xtr.fillna(med).values
        Xte = Xte.fillna(med).values
        ytr = train_df[target].values
        yte = test_df[target].values
        sc = StandardScaler().fit(Xtr)
        Xtr,Xte = sc.transform(Xtr),sc.transform(Xte)
        model = model_factory()
        model.fit(Xtr,ytr)
        pred = model.predict(Xte)
        m = compute_metrics(yte,pred)
        m["n_train"] = int(len(ytr))
        m["n_test"] = int(len(yte))
        results[str(held)] = m
        print(f"[transfer] Held out '{held}': {m}")
    return results