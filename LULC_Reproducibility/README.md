# GEE Scripts for Interpretable LULC Mapping in Shanxi Province

This repository contains the Google Earth Engine (GEE) JavaScript scripts supporting the data preparation, LULC classification, accuracy assessment, feature-ablation experiments, and cross-regional validation reported in **“Interpretable LULC Mapping in the Mountainous Landscape of Shanxi Province: Multi-Source Features and Cross-Regional Validation.”**

## Access

Open the shared repository in the GEE Code Editor:  
<https://code.earthengine.google.com/?accept_repo=users/GEE-Camping/LULC>

The scripts and project assets listed below have been configured for public read access by the repository owner.

## Scripts and workflow

| Script | Description |
|---|---|
| `img_shanxi` | Builds the 2024 Landsat 9 composite for Shanxi and derives spectral indices, SRTM terrain variables, and 5 × 5 GLCM textures; exports the 42-feature image asset. |
| `rf_tuning` | Extracts sample predictors, creates a class-stratified 70/30 training–test split, runs five-fold RF evaluation with configurable settings, and exports the sample tables. |
| `LULC_shanxi` | Trains and evaluates the final Shanxi RF model; exports the LULC map and test predictions for bootstrap analysis. |
| `classifier_shanxi` | Compares Gradient Tree Boost, CART, and SVM using five-fold cross-validation and the independent Shanxi test set. |
| `feature_ablation` | Compares RF classification performance across alternative feature-set configurations. |
| `ESA_contrast` | Reclassifies ESA WorldCover 2021 to the study’s six LULC classes and assesses its accuracy against Shanxi reference samples. |
| `LULC_shaanxi` | Applies the Shanxi-trained RF model in Shaanxi, evaluates it with the Shaanxi external validation samples, and exports the map and validation predictions. |
| `classifier_shaanxi` | Compares alternative classifiers for the Shanxi-to-Shaanxi transfer evaluation. |

The order above follows the main analytical workflow; some scripts use intermediate images or sample assets generated or prepared in earlier steps.

## Data dependencies

**GEE catalog datasets:** Landsat 9 Collection 2 Level-2 Tier 1 (`LANDSAT/LC09/C02/T1_L2`); SRTM (`USGS/SRTMGL1_003`); and ESA WorldCover v200 (2021) (`ESA/WorldCover/v200/2021`).

**Project assets:**

- Boundary: `projects/ee-lingwu462/assets/boundry/China_provinces`
- Feature images: `projects/ee-lingwu462/assets/yes/indexedImg_terrain_Textures`, `.../Img_slim`, and `.../img`
- Main training/test assets: `projects/ee-lingwu462/assets/yes/slim_training` and `.../slim_test`
- Feature-ablation assets: `slim_training_guang`, `slim_test_guang`, `training_set_stratified_indexed`, `test_set_stratified_indexed`, `training_set_stratified_indexed_terrain`, and `test_set_stratified_indexed_terrain` (under `projects/ee-lingwu462/assets/yes/`)
- ESA comparison samples: `projects/ee-lingwu462/assets/yes/slim_test_contrast`

The six-class sample point collections are embedded in the relevant scripts. Project assets are referenced directly by their Earth Engine asset IDs.

## Execution

Open the repository in the GEE Code Editor and run the scripts individually. Start with `img_shanxi` when rebuilding the Shanxi predictor image. Confirm that the required intermediate assets are available before running dependent scripts. Export tasks are submitted through the Code Editor **Tasks** tab; output destinations may need to be changed to locations for which the user has write permission.

## Scope

This repository documents the GEE component of the study. The companion Python workflows for RFECV, feature-correlation analysis, SHAP, PDP/ICE, ALE, and bootstrap-based uncertainty assessment, together with the processed tabular inputs, are provided in the `LULC_Reproducibility_Python/` directory. See its README and requirements file for the Python workflow and execution instructions.
