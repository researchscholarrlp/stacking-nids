"""XGBoost depth check on NSL-KDD. Usage: python nsl_xgb_depth_check.py --data DIR --folds nsl_fold_s0.npy"""
import argparse, numpy as np, pandas as pd, shutil
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--folds", required=True)
a = ap.parse_args()
fam = {}
for n in "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb".split(): fam[n] = "DoS"
for n in "ipsweep nmap portsweep satan mscan saint".split(): fam[n] = "Probe"
for n in ("ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named "
          "snmpgetattack snmpguess xlock xsnoop").split(): fam[n] = "R2L"
for n in "buffer_overflow loadmodule perl rootkit httptunnel ps sqlattack xterm".split(): fam[n] = "U2R"
fam["normal"] = "Normal"
classes = ["DoS", "Normal", "Probe", "R2L", "U2R"]; cid = {c: i for i, c in enumerate(classes)}
tr = pd.read_csv(f"{a.data}/KDDTrain+.txt", header=None)
y = tr[41].str.strip().map(fam).map(cid).values
X = tr.drop(columns=[41, 42]); X.columns = range(X.shape[1])
fold = np.load(f"{a.folds}")
dev = "cuda" if shutil.which("nvidia-smi") else "cpu"
prep = lambda: ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), [1, 2, 3])],
                                 remainder="passthrough", sparse_threshold=0)
def run(depth, folds):
    yt, yp = [], []
    for k in folds:
        te = fold == k; trn = ~te
        pp = prep().fit(X[trn])
        m = XGBClassifier(objective="multi:softprob", n_estimators=300, learning_rate=0.1, max_depth=depth,
                          tree_method="hist", device=dev, random_state=0)
        m.fit(pp.transform(X[trn]), y[trn])
        yp.append(m.predict(pp.transform(X[te]))); yt.append(y[te])
    return f1_score(np.concatenate(yt), np.concatenate(yp), average="macro")
depths = [4, 6, 8, 10, 12, 16, 20]
res = {d: run(d, [0, 1, 2]) for d in depths}
for d in depths: print(f"depth {d}: folds 0-2 macro-F1 {res[d]:.4f}")
best = max(res, key=res.get); print("best:", best)
print("folds 3-9: best", round(run(best, range(3, 10)), 4), "depth 8", round(run(8, range(3, 10)), 4))
