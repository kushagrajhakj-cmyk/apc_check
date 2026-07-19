"""
Retrain the XGBoost + ANN ensemble directly from dummy_plant_data.xlsx.

Unlike train_models.py (which *generates* a fresh synthetic dataset every
run), this script *reads* the existing plant data file as-is, so it can be
pointed at a real plant data extract with the same column schema and will
retrain on whatever is actually in that file.

Rules applied, per request:
  - Text/categorical columns (Catalyst, Grade) are dropped before training
    -- only the 9 numeric process inputs are used as features.
  - Any empty/missing cells in the feature or target columns are filled
    with 0 rather than dropped.

Usage:
    python train_from_plant_data.py [path_to_xlsx]

Outputs (overwrites in the current directory):
    xgb_model.pkl, scaler.pkl, ann_model.pth
"""

import sys
import numpy as np
import pandas as pd
#import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "dummy_plant_data.xlsx"

FEATURES = [
    "Rxn Temp. oC",
    "C2 Pressure kg/cm2",
    "H2/C2",
    "C4/C2",
    "C6/C2",
    "ICA mol %",
    "Al/Ti",
    "Feed rate (kg/h)",
    "Catalyst rate (kg/h)",
]

TARGETS = [
    "MFI @2.16 kg/cm2",
    "Productivity kg PE/g cat",
    "Density g/cc",
]

# Text columns like "Catalyst" and "Grade" are intentionally NOT in
# FEATURES/TARGETS above, so they're simply never selected — this is
# what "don't use text features in training" means in practice here.

# ----------------------------------------------------------
# 1. Load the plant data file as-is (no synthetic regeneration)
# ----------------------------------------------------------

print(f"Loading {DATA_PATH} ...")
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()
print("Raw shape:", df.shape)

missing_cols = [c for c in FEATURES + TARGETS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Expected columns missing from {DATA_PATH}: {missing_cols}")

# Keep only the numeric feature/target columns; drop any text columns
# (Catalyst, Grade, or anything else that isn't a modeled feature/target).
df = df[FEATURES + TARGETS].copy()

# Coerce to numeric in case of stray strings/blanks, then replace any
# empty cells (including ones that failed numeric coercion) with 0.
for c in FEATURES + TARGETS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

n_missing = int(df.isna().sum().sum())
if n_missing:
    print(f"Filling {n_missing} empty cell(s) with 0.")
df = df.fillna(0)

print("Training shape after cleaning:", df.shape)

# ----------------------------------------------------------
# 2. Train XGBoost (multi-output) on 9 features -> 3 targets
# ----------------------------------------------------------

X = df[FEATURES]
Y = df[TARGETS].values

X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

xgb = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
)
xgb.fit(X_tr, Y_tr)

xgb_pred = xgb.predict(X_te)
for i, t in enumerate(TARGETS):
    print(f"XGB  R2 {t}: {r2_score(Y_te[:, i], xgb_pred[:, i]):.4f}")

joblib.dump(xgb, "xgb_model.pkl")

# ----------------------------------------------------------
# 3. Train ANN (9 -> 32 -> 16 -> 3) on scaled inputs
# ----------------------------------------------------------

scaler = StandardScaler().fit(X_tr)
joblib.dump(scaler, "scaler.pkl")

Xtr_t = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
Xte_t = torch.tensor(scaler.transform(X_te), dtype=torch.float32)

y_mean = Y_tr.mean(axis=0)
y_std = Y_tr.std(axis=0)
y_std = np.where(y_std == 0, 1.0, y_std)  # guard against constant targets
Ytr_t = torch.tensor((Y_tr - y_mean) / y_std, dtype=torch.float32)


class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.network(x)


torch.manual_seed(42)
ann = ANNModel()
opt = torch.optim.Adam(ann.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

n = len(Xtr_t)
for epoch in range(300):
    perm = torch.randperm(n)
    for i in range(0, n, 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        loss = loss_fn(ann(Xtr_t[idx]), Ytr_t[idx])
        loss.backward()
        opt.step()
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            print(f"epoch {epoch+1}: train loss {loss_fn(ann(Xtr_t), Ytr_t):.4f}")

# Fold the target un-scaling into the last linear layer so the app can
# keep feeding it raw (unscaled) targets on the way out:
#   raw = z * y_std + y_mean
with torch.no_grad():
    last = ann.network[-1]
    std_t = torch.tensor(y_std, dtype=torch.float32)
    mean_t = torch.tensor(y_mean, dtype=torch.float32)
    last.weight.mul_(std_t.unsqueeze(1))
    last.bias.mul_(std_t)
    last.bias.add_(mean_t)

ann.eval()
with torch.no_grad():
    ann_pred = ann(Xte_t).numpy()
for i, t in enumerate(TARGETS):
    print(f"ANN  R2 {t}: {r2_score(Y_te[:, i], ann_pred[:, i]):.4f}")

torch.save(ann.state_dict(), "ann_model.pth")
print("Saved xgb_model.pkl, scaler.pkl, ann_model.pth")
