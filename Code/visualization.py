# =============================================================================
# Step 10 — Validation Figures
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score,mean_squared_error

def make_validation_plots(all_results,y_true,predictions_dict,out_dir="mdq_outputs"):
    os.makedirs(out_dir,exist_ok=True)
    sns.set_style("whitegrid")
    n = len(predictions_dict)
    if n == 0:
        print("[plot] No predictions to plot.")
        return
    cols = min(3,n)
    rows = (n+cols-1)//cols
    fig,axes = plt.subplots(rows,cols,figsize=(5*cols,5*rows))
    axes = np.array(axes).reshape(-1)
    for ax,(name,pred) in zip(axes,predictions_dict.items()):
        ax.scatter(y_true,pred,alpha=0.7,edgecolor="k")
        lo = min(y_true.min(),pred.min())
        hi = max(y_true.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5)
        r2 = r2_score(y_true,pred)
        rmse = np.sqrt(mean_squared_error(y_true,pred))
        ax.set_title(f"{name}\nR²={r2:.3f} RMSE={rmse:.2f}")
        ax.set_xlabel("Observed AGB (Mg/ha)")
        ax.set_ylabel("Predicted AGB (Mg/ha)")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"scatter_predictions.png"),dpi=200)
    plt.close()
    df_res = pd.DataFrame(all_results).T.reset_index().rename(columns={"index":"Model"})
    fig,ax = plt.subplots(figsize=(10,0.4*len(df_res)+2))
    ax.axis("off")
    tbl = ax.table(cellText=df_res.round(4).values,colLabels=df_res.columns,loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1,1.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"metrics_table.png"),dpi=200)
    plt.close()
    df_res.to_csv(os.path.join(out_dir,"metrics_table.csv"),index=False)
    print(f"[plot] Saved validation figures to {out_dir}/")

def make_learning_curve_plot(lc_dict,out_dir="mdq_outputs"):
    os.makedirs(out_dir,exist_ok=True)
    fracs = sorted(lc_dict.keys())
    n_plots = [lc_dict[f]["n_plots"] for f in fracs]
    r2_mean = [lc_dict[f]["R2_mean"] for f in fracs]
    r2_std = [lc_dict[f]["R2_std"] for f in fracs]
    rmse_mean = [lc_dict[f]["RMSE_mean"] for f in fracs]
    fig,ax1 = plt.subplots(figsize=(8,5))
    ax1.errorbar(n_plots,r2_mean,yerr=r2_std,marker="o",capsize=4,color="tab:blue",label="R²")
    ax1.set_xlabel("Number of training plots")
    ax1.set_ylabel("R²",color="tab:blue")
    ax1.tick_params(axis="y",labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(n_plots,rmse_mean,marker="s",color="tab:red",label="RMSE")
    ax2.set_ylabel("RMSE (Mg/ha)",color="tab:red")
    ax2.tick_params(axis="y",labelcolor="tab:red")
    plt.title("Limited-Field-Data Learning Curve")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir,"learning_curve.png"),dpi=200)
    plt.close()