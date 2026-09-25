# =============================================================================
# Step 9 — Limited-Field-Data Learning Curves
# =============================================================================
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error

def learning_curve(model_factory,X_train,y_train,X_test,y_test,fractions=(0.10,0.25,0.50,0.75,1.00),repeats=5,seed=42):
    print("\n"+"="*70)
    print("STEP 9: Limited-Field-Data Learning Curves")
    print("="*70)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    summary = {}
    for f in fractions:
        k = max(3,int(f*n))
        r2s,rmses = [],[]
        for r in range(repeats):
            idx = rng.choice(n,size=k,replace=False)
            model = model_factory()
            model.fit(X_train[idx],y_train[idx])
            pred = model.predict(X_test)
            r2s.append(r2_score(y_test,pred))
            rmses.append(np.sqrt(mean_squared_error(y_test,pred)))
        summary[f] = {
            "n_plots":int(k),
            "R2_mean":float(np.mean(r2s)),
            "R2_std":float(np.std(r2s)),
            "RMSE_mean":float(np.mean(rmses)),
            "RMSE_std":float(np.std(rmses)),
        }
        print(f"[lc] frac={f:.2f} (n={k}): R2={summary[f]['R2_mean']:.3f}±{summary[f]['R2_std']:.3f} | RMSE={summary[f]['RMSE_mean']:.2f}")
    return summary