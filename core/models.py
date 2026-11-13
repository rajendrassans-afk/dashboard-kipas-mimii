"""Pelatihan, evaluasi, dan perbandingan model/parameter."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn import metrics as skm

WARN_LOW, WARN_HIGH = 0.30, 0.60   # ambang default status peringatan (dapat diubah di UI)


def make_rf(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt", seed=42):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  class_weight="balanced", n_jobs=-1, random_state=seed)


def make_model(name: str, seed=42, **kw):
    if name == "Random Forest":
        return make_rf(seed=seed, **kw)
    if name == "SVM (RBF)":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=kw.get("C", 1.0), probability=True,
                                                   class_weight="balanced", random_state=seed))
    if name == "k-NN":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=kw.get("k", 5)))
    if name == "Regresi Logistik":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    raise ValueError(name)


def split(X, y, groups=None, mode="acak", test_size=0.25, seed=42):
    """mode 'acak': stratified hold-out. mode 'lintas-id': ID mesin uji tidak muncul di latih."""
    idx = np.arange(len(y))
    if mode == "lintas-id" and groups is not None and len(set(groups)) > 1:
        ids = sorted(set(groups)); rng = np.random.default_rng(seed); rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_size))
        te_ids = set(ids[:n_test]); te = np.array([g in te_ids for g in groups])
        return idx[~te], idx[te]
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def evaluate(y_true, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    out = dict(
        akurasi=skm.accuracy_score(y_true, pred),
        presisi=skm.precision_score(y_true, pred, zero_division=0),
        recall=skm.recall_score(y_true, pred, zero_division=0),
        f1=skm.f1_score(y_true, pred, zero_division=0),
        bal_acc=skm.balanced_accuracy_score(y_true, pred),
    )
    out["auc"] = skm.roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    return out


def fit_eval(model_name, X, y, tr, te, **kw):
    t0 = time.time()
    m = make_model(model_name, **kw).fit(X[tr], y[tr])
    fit_t = time.time() - t0
    proba = m.predict_proba(X[te])[:, 1]
    r = evaluate(y[te], proba); r["waktu_latih_s"] = fit_t
    return m, proba, r


def compare_models(X, y, tr, te, names=("Random Forest", "SVM (RBF)", "k-NN", "Regresi Logistik")):
    rows = []
    for n in names:
        _, _, r = fit_eval(n, X, y, tr, te); rows.append(dict(model=n, **r))
    return pd.DataFrame(rows)


def sweep_rf(X, y, tr, te, grid_n=(10, 50, 100, 200, 400), grid_depth=(None, 5, 10, 20)):
    rows = []
    for n in grid_n:
        for d in grid_depth:
            _, _, r = fit_eval("Random Forest", X, y, tr, te, n_estimators=n, max_depth=d)
            rows.append(dict(n_estimators=n, max_depth="None" if d is None else d, **r))
    return pd.DataFrame(rows)


def status(p_abn: float, lo=WARN_LOW, hi=WARN_HIGH):
    """Peta probabilitas -> status peringatan. Bukan diagnosis penyebab fisik."""
    if p_abn < lo:
        return "NORMAL", "ok"
    if p_abn < hi:
        return "WASPADA", "warn"
    return "ABNORMAL", "err"
