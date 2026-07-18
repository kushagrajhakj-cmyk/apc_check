"""
Regenerate dummy plant data with the gas-phase PE reactor schema and
retrain the XGBoost + ANN ensemble.

Columns:
    Catalyst | Grade | Rxn Temp. oC | C2 Pressure kg/cm2 | H2/C2 | C4/C2 |
    C6/C2 | ICA mol % | Al/Ti | Feed rate (kg/h) | Catalyst rate (kg/h) |
    Productivity kg PE/g cat | MFI @2.16 kg/cm2 | Density g/cc

Inputs (9)  -> Rxn Temp, C2 Pressure, H2/C2, C4/C2, C6/C2, ICA mol %,
               Al/Ti, Feed rate, Catalyst rate
Targets (3) -> MFI, Productivity, Density
"""

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

rng = np.random.default_rng(42)
N = 5000

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

# ----------------------------------------------------------
# 1. Sample the operating window
# ----------------------------------------------------------

rxn_temp = rng.uniform(80, 110, N)          # deg C
c2_press = rng.uniform(15, 25, N)           # kg/cm2
h2_c2 = rng.uniform(0.05, 0.50, N)          # mol ratio
c4_c2 = rng.uniform(0.0, 0.40, N)           # mol ratio
c6_c2 = rng.uniform(0.0, 0.15, N)           # mol ratio
ica = rng.uniform(0.0, 12.0, N)             # mol %
al_ti = rng.uniform(20, 120, N)             # mol ratio
feed_rate = rng.uniform(8000, 12000, N)     # kg/h  (C2 feed)
cat_rate = rng.uniform(5, 15, N)            # kg/h

catalyst = rng.choice(["ZN-100", "ZN-200", "CR-300"], N, p=[0.45, 0.35, 0.20])
cat_activity = np.where(catalyst == "ZN-100", 1.00,
                np.where(catalyst == "ZN-200", 1.15, 0.85))
cat_h2_resp = np.where(catalyst == "CR-300", 0.6, 1.0)  # Cr responds less to H2

# ----------------------------------------------------------
# 2. Physics-inspired responses + noise
# ----------------------------------------------------------

# MFI: log-linear in H2/C2 (chain transfer), mild temp effect
log_mfi = (
    -2.2
    + 7.5 * h2_c2 * cat_h2_resp
    + 0.035 * (rxn_temp - 95)
    + 0.8 * c4_c2                      # comonomer slightly raises MFI
    + rng.normal(0, 0.12, N)
)
mfi = np.exp(log_mfi).clip(0.05, 60)

# Density: comonomer incorporation lowers density
density = (
    0.9615
    - 0.070 * c4_c2
    - 0.145 * c6_c2
    + 0.00045 * np.log(mfi + 0.2)      # weak MFI coupling
    + rng.normal(0, 0.0008, N)
).clip(0.905, 0.965)

# Productivity (kg PE / g cat): pressure & monomer driven, parabolic
# optima in temperature and Al/Ti, ICA improves heat removal
productivity = (
    cat_activity * (
        1.5
        + 0.28 * c2_press
        - 0.004 * (rxn_temp - 96) ** 2
        - 0.00055 * (al_ti - 65) ** 2
        + 0.16 * ica
        + 0.00035 * (feed_rate - 8000) / 10
        - 0.12 * cat_rate
    )
    + rng.normal(0, 0.35, N)
).clip(0.5, 15)

# ----------------------------------------------------------
# 3. Assign product grades from MFI / density windows
# ----------------------------------------------------------

def assign_grade(m, d):
    if d < 0.925:
        return "LL-F1820" if m < 5 else "LL-M2420"
    if d < 0.945:
        return "MD-F3840" if m < 5 else "MD-I4550"
    return "HD-B5502" if m < 5 else "HD-J5960"

grade = [assign_grade(m, d) for m, d in zip(mfi, density)]

df = pd.DataFrame({
    "Catalyst": catalyst,
    "Grade": grade,
    "Rxn Temp. oC": rxn_temp,
    "C2 Pressure kg/cm2": c2_press,
    "H2/C2": h2_c2,
    "C4/C2": c4_c2,
    "C6/C2": c6_c2,
    "ICA mol %": ica,
    "Al/Ti": al_ti,
    "Feed rate (kg/h)": feed_rate,
    "Catalyst rate (kg/h)": cat_rate,
    "Productivity kg PE/g cat": productivity,
    "MFI @2.16 kg/cm2": mfi,
    "Density g/cc": density,
})

df.to_excel("dummy_plant_data.xlsx", index=False)
print("Saved dummy_plant_data.xlsx", df.shape)

# ----------------------------------------------------------
# 4. Train XGBoost (multi-output) on 9 features -> 3 targets
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
# 5. Train ANN (9 -> 32 -> 16 -> 3) on scaled inputs
# ----------------------------------------------------------

scaler = StandardScaler().fit(X_tr)
joblib.dump(scaler, "scaler.pkl")

Xtr_t = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
Xte_t = torch.tensor(scaler.transform(X_te), dtype=torch.float32)

# Scale targets for balanced training, bake the inverse transform into
# the final layer afterwards so the app can keep using raw outputs.
y_mean = Y_tr.mean(axis=0)
y_std = Y_tr.std(axis=0)
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

# Fold the target un-scaling into the last linear layer:
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
