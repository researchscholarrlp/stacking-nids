"""Normal-class weight sweep on saved UNSW-NB15 test probabilities (diagnostic, not tuned).
Usage: python unsw_normal_weight_sweep.py --test-csv UNSW_NB15_testing-set.csv --probs unsw_test_p_x_d16.npy"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser(); ap.add_argument("--test-csv", required=True); ap.add_argument("--probs", required=True)
a = ap.parse_args()
te = pd.read_csv(a.test_csv); classes = sorted(te["attack_cat"].str.strip().unique())
y = te["attack_cat"].str.strip().map({c: i for i, c in enumerate(classes)}).values
N = classes.index("Normal"); p = np.load(a.probs)
print("w macroF1 FPR detection")
for w in [1, 2, 4, 8, 12, 16, 32]:
    q = p.copy(); q[:, N] *= w; pred = q.argmax(1)
    print(w, round(f1_score(y, pred, average="macro"), 4),
          round((pred[y == N] != N).mean(), 4), round((pred[y != N] != N).mean(), 4))
