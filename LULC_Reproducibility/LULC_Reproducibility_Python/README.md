# Python Workflow for Interpretable LULC Mapping in Shanxi Province

This directory contains the Python scripts and processed tabular data used in the study **“Interpretable LULC Mapping in the Mountainous Landscape of Shanxi Province: Multi-Source Features and Cross-Regional Validation.”** It complements the Google Earth Engine (GEE) scripts documented in the parent directory and supports feature selection, feature-correlation analysis, model interpretation, and uncertainty assessment.

## Directory structure

```text
LULC_Reproducibility_Python/
├── README.md
├── requirements.txt
├── Python_Scripts/
│   ├── rfecv.py
│   ├── correlation_matrix.py
│   ├── SHAP_ALE.py
│   └── boostrap.py
└── Processed_Data/
    ├── training_set_stratified.csv
    ├── test_set_stratified.csv
    ├── slim_training19.csv
    ├── slim_test19.csv
    ├── LULC_Test_Predictions_for_Bootstrap.csv
    └── Shaanxi_all_for_Bootstrap.csv
```

## Scripts and analytical workflow

| Script | Function | Main inputs |
|---|---|---|
| `rfecv.py` | Performs RFECV using a Random Forest estimator and five-fold cross-validation. Feature subsets are evaluated with both Accuracy and Macro-F1 scoring. It exports feature-count/score tables and a combined feature-selection figure. | `training_set_stratified.csv`, `test_set_stratified.csv` |
| `correlation_matrix.py` | Calculates Pearson correlations among the 19 RFECV-selected predictors and generates a lower-triangular correlation plot. | `slim_training19.csv` |
| `SHAP_ALE.py` | Fits the 19-feature Random Forest model and generates global and class-specific SHAP summaries, class-wise ALE curves, and PDP/ICE response plots. | `slim_training19.csv`, `slim_test19.csv` |
| `boostrap.py` | Calculates confusion-matrix-based accuracy metrics and class-stratified bootstrap 95% confidence intervals for the Shanxi independent test set and Shaanxi external validation set. The script uses 1,000 bootstrap replicates and random seed 42. | `LULC_Test_Predictions_for_Bootstrap.csv`, `Shaanxi_all_for_Bootstrap.csv` |

The script name `boostrap.py` is retained as supplied in the project files.

## Processed data

The `Processed_Data` directory contains the tabular inputs required by the Python scripts.

| File | Content | Records | Columns |
|---|---|---:|---:|
| `training_set_stratified.csv` | Shanxi training set with 42 candidate predictors, the `landcover` label, and coordinates | 1,433 | 45 |
| `test_set_stratified.csv` | Shanxi independent test set with the same 42 predictors, label, and coordinates | 690 | 45 |
| `slim_training19.csv` | Shanxi training set with the 19 RFECV-selected predictors, label, and coordinates | 1,433 | 22 |
| `slim_test19.csv` | Shanxi independent test set with the same 19 predictors, label, and coordinates | 690 | 22 |
| `LULC_Test_Predictions_for_Bootstrap.csv` | Reference and predicted labels for the Shanxi independent test set, exported from GEE | 690 | 4 |
| `Shaanxi_all_for_Bootstrap.csv` | Reference and predicted labels for the Shaanxi external validation set, exported from GEE | 1,829 | 4 |

For the 19-feature matrices, the predictors are `SR_B1`–`SR_B7`, `NDVI`, `EVI`, `MSAVI`, `MNDWI`, `NDTI`, `BLFEI`, `DBSI`, `MBI`, `slope`, `MNDWI_contrast`, `MNDWI_idm`, and `SWIR1_asm`.

The `landcover` field stores the reference class code:

| Code | LULC class |
|---:|---|
| 1 | Cropland |
| 2 | Forest |
| 3 | Grassland |
| 4 | Water |
| 5 | Built-up land |
| 6 | Unused land |

The two prediction files also contain the GEE-exported fields `system:index` and `.geo`; the bootstrap script uses the `landcover` (reference) and `classification` (predicted) columns.

## Main outputs

The scripts generate the following figures and tables (saved relative to the current working directory):

- `rfecv.py`: `RFE_Analysis_Results_sub12.png`, `rfecv_feature_vs_accuracy.csv`, and `rfecv_feature_vs_macro_f1.csv`.
- `correlation_matrix.py`: `corr_matrix_19features_circle_only.png`.
- `SHAP_ALE.py`: `shap_outputs/shap_summary_dot_all_classes.png`, `shap_outputs/shap_mean_importance_stacked.png`, `shap_outputs/ale_curves_by_class/ale_by_class_top5_grid.png`, and `final_comprehensive_gray_ice/comprehensive_pdp_ice_gray_ice.png`.
- `boostrap.py`: `LULC_Test_Predictions_for_Bootstrap_results.csv` and `Shaanxi_all_for_Bootstrap_results.csv`.

## Requirements

The scripts use Python and the packages listed in `requirements.txt`. The required third-party libraries are NumPy, pandas, Matplotlib, scikit-learn, and SHAP. No Earth Engine Python API is required to run these local analyses because the scripts read the supplied CSV files.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied files do not record the exact package versions used in the original execution environment. Accordingly, `requirements.txt` lists the direct dependencies without assigning unverified version pins. For a fully version-locked environment, record the versions of the validated local environment (for example, with `python -m pip freeze`) before archiving the release.

## Running the scripts

The scripts refer to their input CSV files by filename. Run them with `Processed_Data` as the working directory so that the relative input paths resolve correctly. From the `LULC_Reproducibility_Python` directory, for example:

```bash
cd Processed_Data
python ../Python_Scripts/rfecv.py
python ../Python_Scripts/correlation_matrix.py
python ../Python_Scripts/SHAP_ALE.py
python ../Python_Scripts/boostrap.py
```

Generated tables and figures are written to the current working directory or to output subdirectories created by the relevant script. To preserve the supplied input files unchanged, run the scripts from a separate working directory containing copies of the required CSV inputs.

## Scope and reproducibility notes

This directory documents the Python component of the workflow. The GEE-based image preprocessing, feature construction, classification, and map export scripts are documented in the parent directory. The Python scripts operate on the processed feature matrices and prediction tables provided here; they do not independently recreate the GEE imagery or Earth Engine assets.
