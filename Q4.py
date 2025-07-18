##FPCA and Bsplines for classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


train_data = pd.read_csv('ECG200TRAIN-1', sep=',', header=None)
test_data = pd.read_csv('ECG200TEST-1', sep=',', header=None)


# Map -1 → 0, 1 → 1
Y_train = ((train_data.iloc[:, 0] + 1) // 2).astype(int)
Y_test = ((test_data.iloc[:, 0] + 1) // 2).astype(int)
Y = np.concatenate([Y_train, Y_test])

X_train = train_data.iloc[:, 1:].values
X_test = test_data.iloc[:, 1:].values
X = np.vstack([X_train, X_test])
n_samples, n_points = X.shape
x = np.linspace(0, 1, n_points)

# Stratified Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for i_trg, i_tst in sss.split(X, Y):
    pass
print("Class counts in train:", np.bincount(Y[i_trg]))
print("Class counts in test: ", np.bincount(Y[i_tst]))

# B-spline Basis
def bspline_basis(x, knots, degree=3):
    knots_full = np.concatenate(([knots[0]] * degree, knots, [knots[-1]] * degree))
    basis = np.zeros((len(x), len(knots) + degree - 1))
    for i in range(len(knots) + degree - 1):
        c = np.zeros(len(knots) + degree - 1)
        c[i] = 1
        basis[:, i] = BSpline(knots_full, c, degree)(x)
    return basis

knots = np.linspace(0, 1, 8)
B = bspline_basis(x, knots)
Bcoef = np.linalg.lstsq(B, X.T, rcond=None)[0].T

# RF classification with bspline
clf_bs = RandomForestClassifier()
clf_bs.fit(Bcoef[i_trg], Y[i_trg])
pred_bs = clf_bs.predict(Bcoef[i_tst])

labels_bs = sorted(np.unique(np.concatenate((Y[i_tst], pred_bs))))
conf_df_bs = pd.DataFrame(confusion_matrix(Y[i_tst], pred_bs),
                          index=[f"True {i}" for i in labels_bs],
                          columns=[f"Pred {i}" for i in labels_bs])
print("\nConfusion Matrix - B-spline:")
print(conf_df_bs)

# FPCA
mu = X[i_trg].mean(axis=0)
X_centered = X - mu
Cov = np.cov(X_centered[i_trg].T)
Cov = gaussian_filter(Cov, sigma=3)

eigvals, eigvecs = np.linalg.eigh(Cov)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
valid = eigvals > 1e-12


k = min(2, valid.sum())
PCs = eigvecs[:, :k]
FPC_scores = X_centered.dot(PCs)

# RF classification with FPCA
clf_fpca = RandomForestClassifier()
clf_fpca.fit(FPC_scores[i_trg], Y[i_trg])
pred_fpca = clf_fpca.predict(FPC_scores[i_tst])

labels_fpca = sorted(np.unique(np.concatenate((Y[i_tst], pred_fpca))))
conf_df_fpca = pd.DataFrame(confusion_matrix(Y[i_tst], pred_fpca),
                                index=[f"True {i}" for i in labels_fpca],
                                columns=[f"Pred {i}" for i in labels_fpca])
print("\nConfusion Matrix - FPCA:")
print(conf_df_fpca)

