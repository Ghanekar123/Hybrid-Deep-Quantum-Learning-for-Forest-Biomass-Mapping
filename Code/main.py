# =============================================================================
# MDQ-BioMap Main Pipeline
# =============================================================================
import os
import sys
import json
import time
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import CSV_PATH_FALLBACK
from data_loader import load_and_split
from transformer import train_transformer
from quantum_models import run_quantum_regressors
from classical_models import run_baselines
from transfer_learning import cross_domain_evaluation
from learning_curves import learning_curve
from visualization import make_validation_plots,make_learning_curve_plot
from raster_mapping import map_biomass_raster

def main(args):
    t_start = time.time()
    out_dir = args.out_dir
    os.makedirs(out_dir,exist_ok=True)
    data = load_and_split(args.csv,train_frac=0.70,val_frac=0.15)
    all_results = {}
    all_predictions = {}
    q_models = {}
    b_models = {}
    # Step 5
    tm_model,Z,tm_metrics = train_transformer(
        data,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=True
    )
    all_results["MultimodalTransformer"] = tm_metrics
    all_predictions["MultimodalTransformer"] = Z["y_pred_test"]
    # Step 6
    if not args.skip_quantum:
        q_results,q_models = run_quantum_regressors(
            data,Z,vqr_epochs=args.vqr_epochs,vqr_lr=args.vqr_lr)
        all_results.update(q_results)
        if "QSVR" in q_models:
            pred_s = q_models["QSVR"].predict(Z["Z_test"])
            all_predictions["QSVR"] = pred_s*data["y_std"]+data["y_mean"]
        if "VQR" in q_models:
            pred_s = q_models["VQR"].predict(Z["Z_test"])
            all_predictions["VQR"] = pred_s*data["y_std"]+data["y_mean"]
    else:
        print("\n[skip] Quantum regression skipped by flag.")
    # Step 7
    b_results,b_models = run_baselines(data)
    all_results.update(b_results)
    Xte = data["X_test"]
    if "RandomForest" in b_models:
        all_predictions["RandomForest"] = b_models["RandomForest"].predict(Xte)
    if "XGBoost" in b_models:
        all_predictions["XGBoost"] = b_models["XGBoost"].predict(Xte)
    if "RFE-SVM" in b_models:
        rfe,svm = b_models["RFE-SVM"]
        all_predictions["RFE-SVM"] = svm.predict(Xte[:,rfe.support_])
    # Step 8
    feature_cols_all = [c for c in data["df"].columns if c not in ["agb_mg_ha","plot_id","latitude","longitude"]]
    if args.domain_col and args.domain_col in data["df"].columns:
        def rf_factory():
            return RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
        transfer_results = cross_domain_evaluation(
            data["df"],args.domain_col,feature_cols_all,rf_factory)
        with open(os.path.join(out_dir,"cross_domain.json"),"w") as f:
            json.dump(transfer_results,f,indent=2)
    else:
        print("\n[skip] Cross-domain evaluation skipped (use --domain-col to enable).")
    # Step 9
    def rf_lc_factory():
        return RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
    lc = learning_curve(
        rf_lc_factory,
        data["X_train"],data["y_train_raw"],
        data["X_test"],data["y_test_raw"],
        fractions=(0.10,0.25,0.50,0.75,1.00),
        repeats=5)
    with open(os.path.join(out_dir,"learning_curve.json"),"w") as f:
        json.dump(lc,f,indent=2)
    make_learning_curve_plot(lc,out_dir=out_dir)
    # Step 10
    make_validation_plots(
        all_results,data["y_test_raw"],all_predictions,out_dir=out_dir)
    with open(os.path.join(out_dir,"all_metrics.json"),"w") as f:
        json.dump(all_results,f,indent=2)
    pred_df = pd.DataFrame({"y_true":data["y_test_raw"]})
    for name,pred in all_predictions.items():
        pred_df[f"pred_{name}"] = pred
    pred_df.to_csv(os.path.join(out_dir,"test_predictions.csv"),index=False)
    # Optional raster mapping
    if args.raster_stack:
        map_model = b_models.get("RandomForest",None)
        if args.map_model == "QSVR" and "QSVR" in q_models:
            map_model = q_models["QSVR"]
        if map_model is None:
            print("[map] No model available for mapping.")
        else:
            map_biomass_raster(
                args.raster_stack,
                map_model,
                args.map_out,
                data["preprocessor"],
                data["y_mean"],
                data["y_std"])
    # Final summary
    tm = all_results["MultimodalTransformer"]
    print("\n"+"="*60)
    print("MultimodalTransformer metrics")
    print("="*60)
    print(f"R2    : {tm['R2']:.4f}")
    print(f"RMSE  : {tm['RMSE']:.4f}")
    print(f"MAE   : {tm['MAE']:.4f}")
    print(f"bias  : {tm['bias']:.4f}")
    print(f"rRMSE : {tm['rRMSE']:.4f}")
    print("="*60)
    print(f"\n[done] All outputs saved to {out_dir}/")
    print(f"[done] Total time: {time.time()-t_start:.1f}s")

def parse_args():
    p = argparse.ArgumentParser(description="MDQ-BioMap: Multimodal Deep-Quantum Biomass Mapping")
    p.add_argument("--csv",required=True,help="Path to input CSV")
    p.add_argument("--out-dir",default="mdq_outputs")
    p.add_argument("--domain-col",default=None)
    p.add_argument("--skip-quantum",action="store_true")
    p.add_argument("--raster-stack",default=None)
    p.add_argument("--map-out",default="agb_map.tif")
    p.add_argument("--map-model",default="RandomForest",choices=["RandomForest","QSVR"])
    p.add_argument("--d-model",type=int,default=64)
    p.add_argument("--n-heads",type=int,default=4)
    p.add_argument("--n-layers",type=int,default=2)
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--patience",type=int,default=40)
    p.add_argument("--vqr-epochs",type=int,default=150)
    p.add_argument("--vqr-lr",type=float,default=5e-3)
    return p.parse_args()

if __name__ == "__main__":
    if len(sys.argv)>1 and "--csv" in sys.argv:
        args = parse_args()
    else:
        print("[info] No --csv argument provided. Using CSV_PATH_FALLBACK.")
        class Args:
            csv = CSV_PATH_FALLBACK
            out_dir = "mdq_outputs"
            domain_col = None
            skip_quantum = False
            raster_stack = None
            map_out = "agb_map.tif"
            map_model = "RandomForest"
            d_model = 64
            n_heads = 4
            n_layers = 2
            epochs = 300
            lr = 1e-3
            batch_size = 16
            patience = 40
            vqr_epochs = 150
            vqr_lr = 5e-3
        args = Args()
    main(args)