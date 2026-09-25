# =============================================================================
# MDQ-BioMap Configuration
# =============================================================================
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPTICAL_BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
VEG_INDICES = ["NDVI","EVI","SAVI","NDRE","GNDVI","NDMI","NBR"]
SAR_FEATURES = ["VV_mean","VH_mean","VV_std","VH_std","VV_VH_ratio"]
LIDAR_FEATURES = ["GEDI_canopy_height_m","RH25","RH50","RH75","RH95","RH100","GEDI_cover","GEDI_quality"]
TERRAIN_FEATURES = ["elevation_m","slope_deg","aspect_deg","terrain_roughness","TWI","curvature"]
ENV_FEATURES = ["rainfall_mm","temperature_c","soil_moisture","soil_carbon","landcover_class"]

MODALITIES = {
    "optical": OPTICAL_BANDS + VEG_INDICES,
    "sar": SAR_FEATURES,
    "lidar": LIDAR_FEATURES,
    "terrain": TERRAIN_FEATURES,
    "env": ENV_FEATURES,
}

TARGET = "agb_mg_ha"
ID_COLS = ["plot_id","latitude","longitude"]
CAT_COL = "landcover_class"

N_QUBITS = 6
N_LAYERS = 2

CSV_PATH_FALLBACK = r"C:\Aakashimplement\ER298685_Devita Sudhir Ghanekar\Dataset_B_10000_plots.csv"