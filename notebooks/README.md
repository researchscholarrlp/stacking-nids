# Notebooks

Working notebooks (Google Colab, T4 GPU) that produced the numbers in the paper. Outputs are cleared. They are research logs, not polished scripts: some early cells are exploratory, and a few paths point to a private Google Drive folder (`/content/drive/MyDrive/phd_data/`). The data files and saved `.npy` arrays are not in this repository.

| File | What it does |
|---|---|
| `01_data_folds_base_models_working_log.ipynb` | Duplicate-row check, grouped stratified 10-fold split (`fold_id.npy`), RF / XGBoost / CNN base models with out-of-fold probabilities and test probabilities (seeds 0/1/2), first stacking runs, NSL-KDD folds and base models. Contains exploratory and repeated cells. |
| `02_nsl_kdd_threshold_and_checks.ipynb` | NSL-KDD: class mapping, Normal-weight (threshold) choice on CV, stack checks, grouped bootstrap, per-class scores. |
| `03_unsw_depth16_stacks_tests_figures.ipynb` | UNSW: depth-16 XGBoost tuning on folds 0-2 and check on folds 3-9, final stacks, grouped bootstrap, McNemar + Holm, Normal-weight sweep, per-class tables, Figures 1 and 2, NSL-KDD depth check. |

Run order: 01, then 02 and 03. Standalone versions of the two diagnostics are in `../scripts/`.
