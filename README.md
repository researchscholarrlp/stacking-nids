# Does Stacking Help Network Intrusion Detection? A Leakage-Aware Evaluation on UNSW-NB15 and NSL-KDD

Code for Paper 1 of the PhD work of Rakshitha L P (Department of Mathematics, NMIT, Nitte (Deemed to be University)).
Guide and corresponding author: Dr. Kiran S.

## What the paper tests
Whether stacking Random Forest, XGBoost and a 1D CNN (logistic-regression meta-learner) beats a single tuned XGBoost on
UNSW-NB15 and NSL-KDD, under grouped 10-fold cross-validation (identical feature rows stay in one fold), preprocessing fitted
on training folds only, and an official test set scored once.

## Data (not included)
- UNSW-NB15: official training and testing partitions (UNSW Canberra): https://research.unsw.edu.au/projects/unsw-nb15-dataset
- NSL-KDD: KDDTrain+.txt and KDDTest+.txt (Tavallaee et al., 2009). Source of the copy used: **NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html (KDDTrain+.txt, KDDTest+.txt)**
Put the files in one folder and pass it to the scripts with `--data`.

## Contents
- `notebooks/` - the Colab notebooks that build folds, out-of-fold predictions, test fits, stacks and statistics. **[export and add them]**
- `scripts/nsl_xgb_depth_check.py` - XGBoost depth check on NSL-KDD (training folds 0-2, confirmed on folds 3-9).
- `scripts/unsw_normal_weight_sweep.py` - Normal-class weight sweep on saved UNSW-NB15 test probabilities.
- `scripts/make_figures.py` - Figures 1 and 2 of the paper.
- `results/` - small result tables (CSV).

## Seeds and settings
Fold ids are saved and reused for every model. RF 300 trees, XGBoost 300 trees with learning rate 0.1 (depth 16 on UNSW-NB15,
depth 8 on NSL-KDD, both chosen on training folds only), CNN Conv1D(32,3)-Conv1D(64,3)-GlobalMaxPool-Dense128-Dropout0.3,
Adam 1e-3, batch 512, 60 epochs. Base-model seeds 0, 1, 2 for test fits. XGBoost has no row or column sampling, so its seed does not matter.
GPU and CPU XGBoost differ by about 0.001 macro-F1.

## Notes
The Normal-weight sweeps on the UNSW-NB15 test set are a diagnostic. The weight was not chosen on that test set.

## Citation
**[add after publication]**
