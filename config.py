from pathlib import Path
DATA_DIR = Path("input")  # e.g. Path("/content/drive/MyDrive/Flood_Susceptibility/input")
OUTPUT_DIR = Path("output")
PREDICTORS = {
    "Curvature": "Nadia_curvature_2.tif",
    "DrainageDensity": "Nadia_d_Density_4_resampled.tif",
    "TWI": "Nadia_TWI_5_NoData_toZero_masked.tif",
    "SPI": "Nadia_SPI_4.tif",
    "TRI": "Nadia_TRI_14.tif",
    "Slope": "Nadia_slope_own.tif",
    "CurveNumber": "Nadia_CurveNo_reclassified_modelready.tif",
    "NDVI": "Nadia_NDVI_modelready.tif",
    "RainfallPerRainyDay": "rainfall_utm_mm_rainydays_resampled.tif",
    "DistanceToRiver": "Nadia_distance_to_river_8_modelready.tif",
}
TARGET_FILE = "Nadia_flood_raster_7_reclassed.tif"
TARGET_POSITIVE_VALUES = [1]
TARGET_NEGATIVE_VALUES = [0]
RESOLUTIONS_M = [30, 100, 250, 500]
RANDOM_SEED = 42
TEST_BLOCK_FRACTION = 0.20
SPATIAL_BLOCK_SIZE_M = 5000
CV_SPLITS = 5
QUICK_RUN = False
SHAP_BACKGROUND_N = 80
SHAP_EXPLAIN_N = 300
PYSR_TRAIN_N = 2000
PYSR_ITERATIONS = 10 if QUICK_RUN else 100
PYSR_TIMEOUT_SECONDS = 120 if QUICK_RUN else 1800
PYSR_MAX_SIZE = 20
PYSR_SELECT_K_FEATURES = 6
WRITE_INDIVIDUAL_MAPS = True
PREDICTION_CHUNK_SIZE = 200_000
