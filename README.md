# Nadia Flood Susceptibility ML, SHAP and PySR

Reproducible project template for 10 flood-conditioning factors, four predictor resolutions (30, 100, 250, 500 m), seven ML classifiers (RF, SVM, XGB, KNN, NB, AdaBoost, LightGBM), a spatially cross-fitted stacked classifier, classification metrics, SHAP interpretation, and PySR symbolic surrogate modelling.

## Crucial label-support decision
The flood inventory is assumed to have native 500 m support. The code does **not** replicate a 500 m label across all 30 m pixels. Instead, at every predictor resolution it samples predictors at the centres of valid native inventory cells. The test metrics therefore remain evaluated against the same inventory support at all resolutions. Fine-resolution probability maps are model outputs, not independent fine-resolution validation.

## Input files
Put GeoTIFFs in `input/` or edit `DATA_DIR` in `config.py`.

Predictors (filenames configurable):
- `Nadia_curvature_2.tif`
- `Nadia_d_Density_4_resampled.tif`
- `Nadia_TWI_5_NoData_toZero_masked.tif`
- `Nadia_SPI_4.tif`
- `Nadia_TRI_14.tif`
- `Nadia_slope_own.tif`
- `Nadia_CurveNo_reclassified_modelready.tif`
- `Nadia_NDVI_modelready.tif`
- `rainfall_utm_mm_rainydays_resampled.tif`
- `Nadia_distance_to_river_8_modelready.tif`

Target: `Nadia_flood_raster_7_reclassed.tif`. Default coding is 1=flood and 0=non-flood; verify and edit `TARGET_POSITIVE_VALUES` / `TARGET_NEGATIVE_VALUES` before running.

All rasters need valid CRS and georeferencing. Use a projected CRS with metre units. Predictor rasters are bilinearly resampled; categorical flood labels are not resampled for model training. Bilinear interpolation does not create new information from coarse source data such as IMD gridded rainfall.

## Run
1. Install Python dependencies: `pip install -r requirements.txt`
2. Configure `config.py`.
3. Run `python run_pipeline.py`.
4. `QUICK_RUN=True` provides a short smoke test. Use `False` for final analyses. PySR is computationally intensive and requires Julia setup through PySR.

## Outputs
`output/metrics.csv`, `output/maps/`, `output/shap/`, `output/symbolic/`, `output/models/`, and `output/run_metadata.json`.

## Methods caveats for publication
- A spatial-block holdout is separated before model fitting. Thresholds are chosen using training-only out-of-fold probabilities.
- Stacking uses group-based out-of-fold predictions for training its logistic meta-learner.
- SHAP is explanatory, not causal. Non-tree models use a sampled Kernel SHAP approximation.
- PySR approximates stacked-model probabilities; the discovered equation is a surrogate, not a causal law. Its held-out fidelity is reported separately.
- For rigorous final reporting, inspect spatial split class balance, raster alignment, target coding, package versions, model stability across repeated spatial splits, and uncertainty intervals.
- Archive a versioned GitHub release with Zenodo for a persistent DOI. Do not upload restricted or very large source rasters without permission.
