# =============================================================================
# Step 10 — Wall-to-Wall Biomass Mapping
# =============================================================================
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def map_biomass_raster(feature_stack_path,model,out_path,preprocessor,y_mean,y_std,block_size=256):
    if not HAS_RASTERIO:
        print("[map] rasterio not available. Skipping raster mapping.")
        return
    print(f"\n[map] Writing biomass map: {feature_stack_path} -> {out_path}")
    with rasterio.open(feature_stack_path) as src:
        meta = src.meta.copy()
        meta.update(count=1,dtype="float32")
        with rasterio.open(out_path,"w",**meta) as dst:
            for ji,window in src.block_windows(1):
                block = src.read(window=window)
                b,h,w = block.shape
                flat = block.reshape(b,-1).T
                mask = ~np.isnan(flat).any(axis=1)
                out = np.full(flat.shape[0],np.nan,dtype=np.float32)
                if mask.sum()>0:
                    Xp = preprocessor.transform(flat[mask])
                    pred = model.predict(Xp)*y_std+y_mean
                    out[mask] = pred
                dst.write(out.reshape(h,w),1,window=window)
    print(f"[map] Done: {out_path}")