
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import json
from scipy.optimize import differential_evolution
import plotly.express as px
from sklearn.ensemble import IsolationForest
import os
import uuid
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title=" Advanced Process Control",
    page_icon="🏭",
    layout="wide"
)

# =====================================================
# ANN ARCHITECTURE
# =====================================================

class ANNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(9,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,3)

        )

    def forward(self,x):

        return self.network(x)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

   # rf_model = joblib.load("rf_model.pkl")

    xgb_model = joblib.load("xgb_model.pkl")

    scaler = joblib.load("scaler.pkl")

    ann_model = ANNModel()

    ann_model.load_state_dict(
        torch.load(
            "ann_model.pth",
            map_location="cpu"
        )
    )

    ann_model.eval()

    return xgb_model,ann_model,scaler

xgb_model,ann_model,scaler = load_models()

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    data = pd.read_excel(
        "dummy_plant_data.xlsx"
    )

    # Guard against hidden whitespace in Excel headers (a common source
    # of "column not found" errors that only show up on some environments)
    data.columns = data.columns.astype(str).str.strip()

    return data

df = load_data()

# =====================================================
# OPERATOR FEEDBACK LOG
# =====================================================
# Every setpoint recommendation produced by either optimizer mode is
# logged here the moment it's generated (status "pending"), and the
# operator's accept / modify / reject decision is attached to that same
# row once given. Over time this log is the raw material for refining
# the model against real plant experience rather than historical data
# alone:
#   - "modified" rows, in aggregate, reveal a systematic bias to correct
#     (e.g. operators always trimming Rxn Temp a few degrees below what
#     the model suggests for a given grade transition)
#   - "rejected" rows flag combinations the model badly misjudged
#   - once the actual resulting product quality is recorded against a
#     row (via record_actual_outcome), that row becomes a genuine
#     (input, setpoint-used, outcome) example that can be fed back into
#     retraining with extra weight, since it's plant-validated rather
#     than just historical
#
# Storage is a flat CSV next to the app. That's enough for a
# single-instance deployment; if this ever needs multiple concurrent
# writers, swap load/save below for a real database without touching
# any of the call sites.

FEEDBACK_LOG_PATH = "feedback_log.csv"

FEEDBACK_COLUMNS = [

    "id",
    "timestamp",
    "mode",
    "inputs",
    "model_setpoints",
    "model_predicted_quality",
    "model_predicted_std",
    "operator_decision",
    "operator_setpoints",
    "operator_reason",
    "feedback_timestamp",
    "actual_quality",
    "actual_outcome_timestamp"

]


def init_feedback_log():

    if not os.path.exists(FEEDBACK_LOG_PATH):

        pd.DataFrame(columns=FEEDBACK_COLUMNS).to_csv(
            FEEDBACK_LOG_PATH,
            index=False
        )


def load_feedback_log():

    init_feedback_log()

    log = pd.read_csv(FEEDBACK_LOG_PATH)

    # A freshly-created / edited-by-hand CSV might be missing a column
    # added in a later version of this app — backfill so downstream code
    # can always assume every column exists.
    for col in FEEDBACK_COLUMNS:

        if col not in log.columns:

            log[col] = ""

    return log


def save_recommendation(

    mode,
    inputs,
    model_setpoints,
    model_predicted_quality,
    model_predicted_std=None

):
    """Persist a new recommendation as a 'pending' row and return its id.

    Called once, right when a recommendation is computed — before the
    operator has had a chance to react to it — so every recommendation
    is on record even if the operator never gives feedback.
    """

    log = load_feedback_log()

    rec_id = uuid.uuid4().hex[:12]

    row = {

        "id": rec_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "inputs": json.dumps(inputs),
        "model_setpoints": json.dumps(model_setpoints),
        "model_predicted_quality": json.dumps(model_predicted_quality),
        "model_predicted_std": (
            json.dumps(model_predicted_std)
            if model_predicted_std is not None else ""
        ),
        "operator_decision": "pending",
        "operator_setpoints": "",
        "operator_reason": "",
        "feedback_timestamp": "",
        "actual_quality": "",
        "actual_outcome_timestamp": ""

    }

    log = pd.concat(
        [log, pd.DataFrame([row])],
        ignore_index=True
    )

    log.to_csv(FEEDBACK_LOG_PATH, index=False)

    return rec_id


def update_feedback(rec_id, decision, operator_setpoints, reason):
    """Attach the operator's accept / modify / reject decision, plus
    their own setpoints and reasoning if they changed anything, to an
    existing recommendation row."""

    log = load_feedback_log()

    mask = log["id"] == rec_id

    if not mask.any():

        return False

    log.loc[mask, "operator_decision"] = decision

    log.loc[mask, "operator_setpoints"] = (
        json.dumps(operator_setpoints) if operator_setpoints is not None else ""
    )

    log.loc[mask, "operator_reason"] = reason

    log.loc[mask, "feedback_timestamp"] = datetime.now().isoformat(timespec="seconds")

    log.to_csv(FEEDBACK_LOG_PATH, index=False)

    return True


def record_actual_outcome(rec_id, actual_quality):
    """Attach the real, plant-measured product quality once it's known,
    closing the loop between what was recommended/accepted and what
    actually happened. This is what turns a row into a genuine training
    example rather than just an opinion."""

    log = load_feedback_log()

    mask = log["id"] == rec_id

    if not mask.any():

        return False

    log.loc[mask, "actual_quality"] = json.dumps(actual_quality)

    log.loc[mask, "actual_outcome_timestamp"] = datetime.now().isoformat(timespec="seconds")

    log.to_csv(FEEDBACK_LOG_PATH, index=False)

    return True


def render_feedback_widget(rec_id, widget_key_prefix, model_setpoints):
    """Reusable accept / modify / reject control shown under any
    recommendation. `model_setpoints` is a dict of {display label:
    value} for exactly the setpoints the operator might want to
    override — the same dict that was passed to save_recommendation
    for this rec_id.
    """

    st.markdown("### Operator Feedback")

    st.caption(
        "Accept these setpoints, or override them based on plant "
        "experience. Every decision is logged and used to refine the "
        "model over time — the reason you give matters as much as the "
        "number."
    )

    decision_label = st.radio(

        "Your decision",

        ["Accept as-is", "Accept with modification", "Reject"],

        key=f"{widget_key_prefix}_decision_{rec_id}",

        horizontal=True

    )

    operator_setpoints = None

    if decision_label == "Accept with modification":

        st.caption("Enter the setpoints you'll actually use on the plant:")

        items = list(model_setpoints.items())

        half = (len(items) + 1) // 2

        oc1, oc2 = st.columns(2)

        operator_setpoints = {}

        for i, (label, value) in enumerate(items):

            col = oc1 if i < half else oc2

            with col:

                operator_setpoints[label] = st.number_input(

                    label,

                    value=float(value),

                    key=f"{widget_key_prefix}_opval_{rec_id}_{i}",

                    format="%.4f"

                )

    reason = ""

    if decision_label in ("Accept with modification", "Reject"):

        reason = st.text_area(

            "Reason, based on plant experience (required)",

            key=f"{widget_key_prefix}_reason_{rec_id}",

            help="E.g. 'this fouls the exchanger above 105C on this "
                 "grade' or 'catalyst rate too aggressive for current "
                 "batch of catalyst'. This is what eventually teaches "
                 "the model what it's getting wrong."

        )

    submitted_key = f"{widget_key_prefix}_submitted_{rec_id}"

    if st.button("Submit Feedback", key=f"{widget_key_prefix}_submit_{rec_id}"):

        if decision_label != "Accept as-is" and not reason.strip():

            st.error(
                "Please add a reason before submitting — it's what "
                "makes this feedback useful later on."
            )

        else:

            decision_map = {
                "Accept as-is": "accepted",
                "Accept with modification": "modified",
                "Reject": "rejected"
            }

            update_feedback(

                rec_id,

                decision_map[decision_label],

                operator_setpoints,

                reason.strip()

            )

            st.session_state[submitted_key] = True

    if st.session_state.get(submitted_key):

        st.success(f"Feedback recorded for recommendation `{rec_id}`.")


# =====================================================
# TIME-SERIES ANOMALY DETECTION (rolling-window Isolation Forest)
# =====================================================
# Instead of feeding raw single-row snapshots into Isolation Forest
# (which only catches instantaneous outliers), we build rolling-window
# statistics (mean + std over a sliding window) per variable. This lets
# the model catch temporal pattern shifts / drifts, not just one-off spikes.

feature_names = [

    "Rxn Temp. oC",
    "C2 Pressure kg/cm2",
    "H2/C2",
    "C4/C2",
    "C6/C2",
    "ICA mol %",
    "Al/Ti",
    "Feed rate (kg/h)",
    "Catalyst rate (kg/h)"

]

# Product-quality columns aren't guaranteed to exist under these exact
# names in every data file, so detect them rather than hardcoding —
# this is also what caused the earlier KeyError.
target_names = [
    c for c in [
        "MFI @2.16 kg/cm2",
        "Productivity kg PE/g cat",
        "Density g/cc"
    ] if c in df.columns
]


def compute_rolling_features(df, feature_names, window):

    roll_mean = df[feature_names].rolling(window).mean()
    roll_std = df[feature_names].rolling(window).std()

    roll_mean.columns = [f"{c}_roll_mean" for c in feature_names]
    roll_std.columns = [f"{c}_roll_std" for c in feature_names]

    feats = pd.concat([roll_mean, roll_std], axis=1)

    return feats


@st.cache_resource
def train_ts_anomaly_model(df, window, contamination):

    feats = compute_rolling_features(df, feature_names, window)
    feats = feats.dropna()

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    model.fit(feats)

    return model, feats


@st.cache_data
def get_feature_correlations(df, feature_names):

    return df[feature_names].corr()


def run_ts_anomaly_detection(df, window=20, contamination=0.03):

    model, feats = train_ts_anomaly_model(df, window, contamination)

    scores = model.decision_function(feats)
    preds = model.predict(feats)

    results = df.loc[feats.index].copy()
    results["anomaly_score"] = scores
    results["is_anomaly"] = preds == -1
    results["health"] = ((scores + 0.5) * 100).clip(0, 100)

    # Per-variable deviation, in rolling-window std units, at every point.
    # This is what lets us explain *why* a point was flagged: whichever
    # variable(s) have the largest |z| are the ones driving the anomaly.
    for c in feature_names:

        mean_col = feats.loc[results.index, f"{c}_roll_mean"]
        std_col = feats.loc[results.index, f"{c}_roll_std"].replace(0, np.nan)

        results[f"{c}_zscore"] = (results[c] - mean_col) / std_col

    return results

# =====================================================
# ENSEMBLE PREDICTION
# =====================================================

def ensemble_predict(input_vector):

    X_df = pd.DataFrame(
        [input_vector],
        columns=feature_names
    )

    

    xgb_pred = xgb_model.predict(
        X_df
    )[0]

    scaled = scaler.transform(
        X_df
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        ann_pred = ann_model(
            tensor
        ).numpy()[0]

    preds = np.array([
        
        xgb_pred,
        ann_pred
    ])

    mean_pred = preds.mean(axis=0)

    std_pred = preds.std(axis=0)

    return mean_pred,std_pred,preds


def ensemble_predict_batch(X):

    # Same ensemble math as ensemble_predict() above (average of the XGB
    # and ANN predictions), just evaluated for many rows at once instead
    # of one row per Python call. This exists purely to make the
    # differential_evolution search in patent_grade_transition_setpoints
    # fast enough to finish (it evaluates thousands of candidate points,
    # and calling ensemble_predict() once per candidate meant re-building
    # a DataFrame and re-running both models thousands of times). It does
    # not change what gets predicted for a given input — for a single
    # row it returns exactly what ensemble_predict() returns as mean_pred.

    X_df = pd.DataFrame(
        X,
        columns=feature_names
    )

    xgb_pred = xgb_model.predict(X_df)

    scaled = scaler.transform(X_df)

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        ann_pred = ann_model(tensor).numpy()

    mean_pred = (np.asarray(xgb_pred) + np.asarray(ann_pred)) / 2.0

    return mean_pred

# =====================================================
# OPTIMIZER
# =====================================================

def optimize_process(

    feed_rate,
    target_mfi,
    target_productivity,
    target_density

):
    # C2 pressure is now manipulated by the optimizer itself (it used to
    # be a fixed input), alongside Rxn Temp, H2/C2, C4/C2, C6/C2, ICA,
    # Al/Ti and Catalyst rate — i.e. every lever the process actually has,
    # per the "manipulate H2/C2, Al/Ti, comonomer ratios, Rxn T and C2
    # pressure" requirement.

    def objective(x):

        rxn_temp = x[0]
        c2_pressure = x[1]
        h2_c2 = x[2]
        c4_c2 = x[3]
        c6_c2 = x[4]
        ica = x[5]
        al_ti = x[6]
        cat_rate = x[7]

        input_vector = [

            rxn_temp,
            c2_pressure,
            h2_c2,
            c4_c2,
            c6_c2,
            ica,
            al_ti,
            feed_rate,
            cat_rate

        ]

        pred,std,_ = ensemble_predict(
            input_vector
        )

        # Relative errors so MFI (~10), productivity (~7) and
        # density (~0.95) are weighted comparably.
        loss = (

            ((pred[0]-target_mfi)/max(target_mfi,1e-6))**2

            +

            ((pred[1]-target_productivity)/max(target_productivity,1e-6))**2

            +

            ((pred[2]-target_density)/max(target_density,1e-6))**2

            +

            0.01*np.sum(std/np.abs(pred).clip(1e-6))

        )

        return loss

    bounds = [

        (80,110),      # Rxn Temp. oC
        (15,25),       # C2 Pressure kg/cm2
        (0.05,0.50),   # H2/C2
        (0.0,0.40),    # C4/C2
        (0.0,0.15),    # C6/C2
        (0.0,12.0),    # ICA mol %
        (20,120),      # Al/Ti
        (5,15)         # Catalyst rate (kg/h)

    ]

    result = differential_evolution(

        objective,

        bounds,

        maxiter=50,

        popsize=10,

        seed=42

    )

    return result


# =====================================================
# PATENT-BASED (STATISTICAL) GRADE-TRANSITION SETPOINTS
# =====================================================
# Setpoint calculator following the grade-transition logic described in
# US 5,627,242. The patent fixes two things: (1) which DIRECTION each
# setpoint moves (e.g. "above Product 2's temp if MI is increasing, else
# below"), and (2) the LEGAL RANGE it's allowed to move within (e.g.
# "1-15 C"). It does not say exactly where in that range to land, and it
# says nothing at all about H2/C2, comonomer ratios, ICA or Al/Ti — those
# are free process levers the patent doesn't touch.
#
# So this does a single joint search: temp_delta and pressure_delta are
# bounded and directed exactly as the patent specifies, while H2/C2,
# C4/C2, C6/C2, ICA, Al/Ti and catalyst rate are free to roam their full
# normal operating range. The trained ML ensemble scores every candidate
# combination, and the loss is weighted so MFI and density (the two
# properties that actually define whether a grade is "in spec") dominate
# the search, while productivity is only weighted lightly — it's allowed
# to be traded off, not protected.

PSIG_TO_KGCM2 = 0.0703069

# Relative-error weights in the search objective. MFI and density define
# grade spec; productivity is a cost/throughput concern that's explicitly
# allowed to be compromised here, so it gets a much smaller weight.
MFI_WEIGHT = 1.0
DENSITY_WEIGHT = 1.0
PRODUCTIVITY_WEIGHT = 0.05

def patent_grade_transition_setpoints(

    p1_temp,
    p1_mi,
    p1_pressure,
    p2_temp,
    p2_mi,
    p2_density,
    feed_rate,
    target_productivity

):

    # Step 1: initial reaction-temperature setpoint change — drop
    # immediately to Product 2's temperature only if it is lower than
    # Product 1's; otherwise leave the setpoint at Product 1's value
    # for now (it will be trimmed precisely in step 3).
    temp_setpoint_initial = p2_temp if p2_temp < p1_temp else p1_temp

    # Steps 2-4 all depend on whether Product 2's MI is higher or lower
    # than Product 1's MI ("... and vice versa" in the patent text).
    mi_increasing = p2_mi > p1_mi

    # x = [temp_delta, pressure_delta, h2_c2, c4_c2, c6_c2, ica, al_ti, cat_rate]
    # The first two are patent-bounded (direction fixed by mi_increasing);
    # the rest are free levers searched over their normal operating range
    # (same ranges the data-driven optimizer uses).
    bounds = [

        (1.0, 15.0),     # temp_delta (patent: 1-15 C)
        (1.0, 25.0),     # pressure_delta (patent: 1-25 psig)
        (0.05, 0.50),    # H2/C2
        (0.0, 0.40),     # C4/C2
        (0.0, 0.15),     # C6/C2
        (0.0, 12.0),     # ICA mol %
        (20.0, 120.0),   # Al/Ti
        (5.0, 15.0)      # Catalyst rate (kg/h)

    ]

    def objective(x):

        temp_delta, pressure_delta, h2_c2, c4_c2, c6_c2, ica, al_ti, cat_rate = x

        temp_refined = (
            p2_temp + temp_delta if mi_increasing else p2_temp - temp_delta
        )
        pressure_sp = (
            p1_pressure - pressure_delta if mi_increasing else p1_pressure + pressure_delta
        )

        input_vector = [

            temp_refined,
            pressure_sp * PSIG_TO_KGCM2,
            h2_c2,
            c4_c2,
            c6_c2,
            ica,
            al_ti,
            feed_rate,
            cat_rate

        ]

        pred,_,_ = ensemble_predict(input_vector)
        mfi_pred, prod_pred, density_pred = pred

        return (

            MFI_WEIGHT * ((mfi_pred - p2_mi) / max(p2_mi,1e-6)) ** 2

            +

            DENSITY_WEIGHT * ((density_pred - p2_density) / max(p2_density,1e-6)) ** 2

            +

            PRODUCTIVITY_WEIGHT * ((prod_pred - target_productivity) / max(target_productivity,1e-6)) ** 2

        )

    def objective_vectorized(x_pop):

        # Identical formula to objective() above, just evaluated for the
        # whole differential_evolution population in one shot (via
        # ensemble_predict_batch) instead of one candidate per Python
        # call. This is what makes the search finish in seconds instead
        # of hanging — nothing about the bounds, weights, or direction
        # rules changes; scipy requires updating='deferred' whenever a
        # vectorized objective is used, hence that setting below.

        x_pop = np.asarray(x_pop)

        temp_delta = x_pop[0]
        pressure_delta = x_pop[1]
        h2_c2 = x_pop[2]
        c4_c2 = x_pop[3]
        c6_c2 = x_pop[4]
        ica = x_pop[5]
        al_ti = x_pop[6]
        cat_rate = x_pop[7]

        n_candidates = x_pop.shape[1]

        if mi_increasing:
            temp_refined = p2_temp + temp_delta
            pressure_sp = p1_pressure - pressure_delta
        else:
            temp_refined = p2_temp - temp_delta
            pressure_sp = p1_pressure + pressure_delta

        X = np.column_stack([

            temp_refined,
            pressure_sp * PSIG_TO_KGCM2,
            h2_c2,
            c4_c2,
            c6_c2,
            ica,
            al_ti,
            np.full(n_candidates, feed_rate),
            cat_rate

        ])

        mean_pred = ensemble_predict_batch(X)

        mfi_pred = mean_pred[:, 0]
        prod_pred = mean_pred[:, 1]
        density_pred = mean_pred[:, 2]

        return (

            MFI_WEIGHT * ((mfi_pred - p2_mi) / max(p2_mi,1e-6)) ** 2

            +

            DENSITY_WEIGHT * ((density_pred - p2_density) / max(p2_density,1e-6)) ** 2

            +

            PRODUCTIVITY_WEIGHT * ((prod_pred - target_productivity) / max(target_productivity,1e-6)) ** 2

        )

    result = differential_evolution(

        objective_vectorized,

        bounds,

        maxiter=60,

        popsize=15,

        seed=42,

        vectorized=True,

        updating="deferred"

    )

    (
        best_temp_delta,
        best_pressure_delta,
        best_h2_c2,
        best_c4_c2,
        best_c6_c2,
        best_ica,
        best_al_ti,
        best_cat_rate

    ) = result.x

    # Step 3: refined reaction-temperature setpoint, 1-15 C above the
    # desired Product 2 temperature if MI is increasing, else below —
    # with the magnitude picked by the search above.
    temp_setpoint_refined = (
        p2_temp + best_temp_delta if mi_increasing else p2_temp - best_temp_delta
    )

    # Step 4: rate-limiting reactant partial pressure setpoint, 1-25
    # psig below Product 1's pressure if MI is increasing, else above.
    pressure_setpoint = (
        p1_pressure - best_pressure_delta if mi_increasing else p1_pressure + best_pressure_delta
    )

    # What does the model think this whole combination will actually
    # produce?
    final_input = [

        temp_setpoint_refined,
        pressure_setpoint * PSIG_TO_KGCM2,
        best_h2_c2,
        best_c4_c2,
        best_c6_c2,
        best_ica,
        best_al_ti,
        feed_rate,
        best_cat_rate

    ]

    pred, std, _ = ensemble_predict(final_input)
    predicted_mfi, predicted_productivity, predicted_density = pred

    # Step 2: melt index setpoint. The patent allows 0-150% higher, or
    # 0-70% lower, than the Product 2 target — i.e. a legal setpoint
    # range of [p2_mi, 2.5*p2_mi] if MI is increasing, or
    # [0.3*p2_mi, p2_mi] if decreasing. Rather than asking the user to
    # pick a point in that range, we set it to whatever the model
    # predicts the optimized combination will actually deliver, clipped
    # into the legal range (so it's always a defensible, patent-
    # compliant value, but driven by the search rather than a guess).
    if mi_increasing:
        mi_low, mi_high = p2_mi, p2_mi * 2.5
    else:
        mi_low, mi_high = p2_mi * 0.3, p2_mi

    mi_setpoint = min(max(predicted_mfi, mi_low), mi_high)
    mi_pct = (mi_setpoint / p2_mi - 1) * 100.0

    return {

        "temp_setpoint_initial": temp_setpoint_initial,
        "mi_setpoint": mi_setpoint,
        "mi_pct": mi_pct,
        "temp_setpoint_refined": temp_setpoint_refined,
        "temp_delta": best_temp_delta,
        "pressure_setpoint": pressure_setpoint,
        "pressure_delta": best_pressure_delta,
        "mi_increasing": mi_increasing,
        "h2_c2": best_h2_c2,
        "c4_c2": best_c4_c2,
        "c6_c2": best_c6_c2,
        "ica": best_ica,
        "al_ti": best_al_ti,
        "cat_rate": best_cat_rate,
        "predicted_mfi": predicted_mfi,
        "predicted_productivity": predicted_productivity,
        "predicted_density": predicted_density,
        "predicted_std": std

    }


# =====================================================
# HEADER
# =====================================================

st.title(
    "PetChem Advanced Process Control Dashboard"
)

st.markdown(
    "ML Models Ensemble"
)

# =====================================================
# SIDEBAR
# =====================================================


st.sidebar.markdown("---")

st.sidebar.header(
    "Developed by Team MacroMinds, Petchem lab"
)

st.sidebar.markdown("---")


# =====================================================
# TABS
# =====================================================
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(

    [

        "Live PFD",

        "Historical Data",

        "Optimizer",

        "Model diagnostics",

        "Anomaly Detection",

        "Feedback & Refinement"


    ]

)
# =====================================================
# HISTORICAL DATA
# =====================================================

with tab1:



    current = df.iloc[-1]

    defaults = {
        "feed_rate": float(current["Feed rate (kg/h)"]),
        "c2_pressure": float(current["C2 Pressure kg/cm2"]),
        "rxn_temp": float(current["Rxn Temp. oC"]),
        "h2_c2": float(current["H2/C2"]),
        "c4_c2": float(current["C4/C2"]),
        "c6_c2": float(current["C6/C2"]),
        "ica_mol": float(current["ICA mol %"]),
        "al_ti": float(current["Al/Ti"]),
        "cat_rate": float(current["Catalyst rate (kg/h)"]),
        "mfi_pred": float(current["MFI @2.16 kg/cm2"]),
        "prod_pred": float(current["Productivity kg PE/g cat"]),
        "dens_pred": float(current["Density g/cc"])
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


    st.markdown(
        f"""
<div style="
width:100%;
height:650px;
position:relative;
border:2px solid #555;
border-radius:12px;
background:#fafafa;
overflow:hidden;
font-family:Arial;
">

<!-- ================= C2 FEED (flow + pressure) ================= -->
<!-- horizontal line from x=40 to x=420 (reactor left edge) -->

<div style="
position:absolute;
left:40px;
top:300px;
width:380px;
border-top:4px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:406px;
top:295px;
width:0;height:0;
border-top:7px solid transparent;
border-bottom:7px solid transparent;
border-left:14px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:45px;
top:235px;
color:#0D47A1;
font-size:14px;
">
<b>C2 FEED</b><br>
Flow = {st.session_state.feed_rate:.0f} kg/h<br>
P = {st.session_state.c2_pressure:.1f} kg/cm&sup2;
</div>

<!-- ================= COMONOMER (C4/C2, C6/C2 ratios) ================= -->
<!-- horizontal line from x=40 to x=420 (reactor left edge) -->
<!-- Same start point (x=40) and width as the C2 FEED line above, so both -->
<!-- feed lines are the same length. Its label keeps the same 15px gap -->
<!-- below its line that every other stream label keeps from its line. -->

<div style="
position:absolute;
left:40px;
top:380px;
width:380px;
border-top:3px solid green;
"></div>

<div style="
position:absolute;
left:406px;
top:374px;
width:0;height:0;
border-top:7px solid transparent;
border-bottom:7px solid transparent;
border-left:14px solid green;
"></div>

<div style="
position:absolute;
left:45px;
top:395px;
color:green;
font-size:13px;
">
<b>COMONOMER</b><br>
C4/C2 = {st.session_state.c4_c2:.3f}<br>
C6/C2 = {st.session_state.c6_c2:.3f}
</div>

<!-- ================= H2 (ratio to C2) ================= -->
<!-- vertical line ending at y=110 (reactor top edge) -->

<div style="
position:absolute;
left:468px;
top:55px;
width:4px;
height:55px;
background:#1976D2;
"></div>

<div style="
position:absolute;
left:463px;
top:96px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-top:14px solid #1976D2;
"></div>

<div style="
position:absolute;
left:400px;
top:6px;
color:#1565C0;
text-align:center;
font-size:14px;
">
<b>H&#8322;</b><br>
H2/C2 = {st.session_state.h2_c2:.3f}
</div>

<!-- ================= ICA (induced condensing agent) ================= -->
<!-- vertical line ending at y=110 (reactor top edge) -->

<div style="
position:absolute;
left:548px;
top:55px;
width:4px;
height:55px;
background:#8E24AA;
"></div>

<div style="
position:absolute;
left:543px;
top:96px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-top:14px solid #8E24AA;
"></div>

<div style="
position:absolute;
left:530px;
top:6px;
color:#6A1B9A;
text-align:center;
font-size:14px;
">
<b>ICA</b><br>
{st.session_state.ica_mol:.1f} mol %
</div>

<!-- ================= REACTOR ================= -->

<div style="
position:absolute;
left:420px;
top:110px;
width:180px;
height:360px;
border:4px solid #1976D2;
border-radius:25px;
overflow:hidden;
background:#FAFAFA;
display:flex;
align-items:center;
justify-content:center;
text-align:center;
">

<div style="
color:#0D47A1;
font-weight:bold;
font-size:15px;
">
REACTOR
<br><br>
Rxn Temperature<br>
{st.session_state.rxn_temp:.1f} &deg;C
</div>

</div>

<!-- ================= CATALYST ================= -->
<!-- vertical line from y=470 (reactor bottom edge) down -->

<div style="
position:absolute;
left:470px;
top:470px;
width:4px;
height:70px;
background:red;
"></div>

<div style="
position:absolute;
left:465px;
top:470px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-bottom:14px solid red;
"></div>

<div style="
position:absolute;
left:395px;
top:555px;
color:red;
text-align:center;
font-size:13px;
">
<b>CATALYST</b><br>
{st.session_state.cat_rate:.2f} kg/h
</div>

<!-- ================= COCATALYST (Al/Ti) ================= -->
<!-- vertical line from y=470 (reactor bottom edge) down -->

<div style="
position:absolute;
left:550px;
top:470px;
width:4px;
height:70px;
background:orange;
"></div>

<div style="
position:absolute;
left:545px;
top:470px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-bottom:14px solid orange;
"></div>

<div style="
position:absolute;
left:525px;
top:555px;
color:orange;
text-align:center;
font-size:13px;
">
<b>COCATALYST</b><br>
Al/Ti = {st.session_state.al_ti:.1f}
</div>

<!-- ================= PRODUCT ================= -->
<!-- horizontal line from x=600 (reactor right edge) -->

<div style="
position:absolute;
left:600px;
top:290px;
width:220px;
border-top:4px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:806px;
top:285px;
width:0;height:0;
border-top:7px solid transparent;
border-bottom:7px solid transparent;
border-left:14px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:640px;
top:170px;
color:#0D47A1;
font-size:14px;
">
<b>PRODUCT</b><br>

MFI @2.16 = {st.session_state.mfi_pred:.2f}<br>

Productivity = {st.session_state.prod_pred:.2f} kg PE/g cat<br>

Density = {st.session_state.dens_pred:.4f} g/cc

</div>

</div>
""",
unsafe_allow_html=True
)


    st.markdown("### Process Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.number_input(
         "C2 Feed Rate (kg/h)",
          key="feed_rate"
         )

        st.number_input(
         "C2 Pressure (kg/cm²)",
         key="c2_pressure"
        )

        st.number_input(
            "Rxn Temperature (°C)",
            key="rxn_temp"
        )

    with c2:

        st.number_input(
            "H2/C2 ratio",
            key="h2_c2",
            format="%.3f"
        )

        st.number_input(
            "C4/C2 ratio",
            key="c4_c2",
            format="%.3f"
        )

        st.number_input(
            "C6/C2 ratio",
            key="c6_c2",
            format="%.3f"
        )

    with c3:

        st.number_input(
            "ICA (mol %)",
            key="ica_mol"
        )

        st.number_input(
            "Al/Ti ratio",
            key="al_ti"
        )

        st.number_input(
            "Catalyst Rate (kg/h)",
            key="cat_rate"
        )



    predict_button = st.button(
        "Predict Product Quality",
         use_container_width=True
        )

    if predict_button:

        input_vector = [

            st.session_state.rxn_temp,
            st.session_state.c2_pressure,
            st.session_state.h2_c2,
            st.session_state.c4_c2,
            st.session_state.c6_c2,
            st.session_state.ica_mol,
            st.session_state.al_ti,
            st.session_state.feed_rate,
            st.session_state.cat_rate

        ]

        pred, std, preds = ensemble_predict(
            input_vector
        )

        st.session_state.mfi_pred = float(pred[0])
        st.session_state.prod_pred = float(pred[1])
        st.session_state.dens_pred = float(pred[2])

        confidence = np.exp(
            -np.mean(std/np.abs(pred).clip(1e-6))
        ) * 100

        st.success(
            f"Prediction Completed | Confidence = {confidence:.1f}%"
        )

        




with tab2:

    st.subheader("Historical Plant Data")

    st.dataframe(df.head(50))

    st.markdown("---")

    st.subheader("Interactive Plot Builder")

    plot_type = st.selectbox(

        "Select Plot Type",

        [
            "Trend Plot",
            "Scatter Plot",
            "Histogram",
            "3D Scatter Plot",
            "Correlation Heatmap"
        ]

    )

    columns = df.columns.tolist()

    # ==========================================
    # TREND PLOT
    # ==========================================

    if plot_type == "Trend Plot":

        selected_column = st.selectbox(

            "Select Parameter",

            columns

        )

        fig = px.line(

            df,

            y=selected_column,

            title=f"{selected_column} Trend"

        )

        fig.update_layout(

            xaxis_title="Sample Number",

            yaxis_title=selected_column

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ==========================================
    # SCATTER PLOT
    # ==========================================

    elif plot_type == "Scatter Plot":

        col1,col2 = st.columns(2)

        with col1:

            x_var = st.selectbox(

                "X Axis",

                columns,

                key="scatter_x"

            )

        with col2:

            y_var = st.selectbox(

                "Y Axis",

                columns,

                index=1,

                key="scatter_y"

            )

        add_trendline = st.checkbox(

            "Add Trendline",

            value=True

        )

        fig = px.scatter(

            df,

            x=x_var,

            y=y_var,

            title=f"{x_var} vs {y_var}"

        )

        # NOTE: we deliberately do NOT use plotly express's built-in
        # trendline="ols" here, since that requires the "statsmodels"
        # package which isn't installed in this environment and throws
        # a ModuleNotFoundError at render time. Instead we fit a simple
        # OLS line ourselves with numpy and add it as an extra trace,
        # which needs nothing beyond numpy/pandas.
        if add_trendline:

            x_num = pd.to_numeric(df[x_var], errors="coerce")
            y_num = pd.to_numeric(df[y_var], errors="coerce")

            mask = x_num.notna() & y_num.notna()

            if mask.sum() >= 2 and x_num[mask].nunique() > 1:

                x_vals = x_num[mask].to_numpy()
                y_vals = y_num[mask].to_numpy()

                slope, intercept = np.polyfit(x_vals, y_vals, 1)

                y_hat = slope * x_vals + intercept
                ss_res = np.sum((y_vals - y_hat) ** 2)
                ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_line = slope * x_line + intercept

                fig.add_scatter(

                    x=x_line,

                    y=y_line,

                    mode="lines",

                    name=f"OLS fit (R\u00b2={r2:.3f})",

                    line=dict(color="red", dash="dash")

                )

            else:

                st.warning(

                    "Not enough numeric data in the selected columns to "
                    "fit a trendline."

                )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ==========================================
    # HISTOGRAM
    # ==========================================

    elif plot_type == "Histogram":

        selected_column = st.selectbox(

            "Select Parameter",

            columns,

            key="hist"

        )

        fig = px.histogram(

            df,

            x=selected_column,

            nbins=30,

            title=f"{selected_column} Distribution"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ==========================================
    # 3D SCATTER
    # ==========================================

    elif plot_type == "3D Scatter Plot":

        numeric_cols = df.select_dtypes(

            include=np.number

        ).columns.tolist()

        col1,col2,col3,col4 = st.columns(4)

        with col1:

            x_axis = st.selectbox(

                "X Axis",

                numeric_cols,

                key="x3d"

            )

        with col2:

            y_axis = st.selectbox(

                "Y Axis",

                numeric_cols,

                index=min(1,len(numeric_cols)-1),

                key="y3d"

            )

        with col3:

            z_axis = st.selectbox(

                "Z Axis",

                numeric_cols,

                index=min(2,len(numeric_cols)-1),

                key="z3d"

            )

        with col4:

            color_axis = st.selectbox(

                "Color By",

                numeric_cols,

                index=min(3,len(numeric_cols)-1),

                key="color3d"

            )

        fig = px.scatter_3d(

            df,

            x=x_axis,

            y=y_axis,

            z=z_axis,

            color=color_axis,

            title="Interactive Operating Window"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    # ==========================================
    # CORRELATION HEATMAP
    # ==========================================

    elif plot_type == "Correlation Heatmap":

        numeric_df = df.select_dtypes(

            include=np.number

        )

        corr = numeric_df.corr()

        fig = px.imshow(

            corr,

            text_auto=True,

            aspect="auto",

            title="Correlation Matrix"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )
# =====================================================
# OPTIMIZER
# =====================================================


with tab3:

    st.subheader(
        "APC Optimizer"
    )

    optimizer_mode = st.radio(

        "Optimization Mode",

        [
            "Based on Data (ML Optimizer)",
            "Based on Data + Knowledge base"
        ],

        key="optimizer_mode",

        horizontal=True

    )

    st.markdown("---")

    # ==========================================
    # MODE 1 — DATA-DRIVEN ML OPTIMIZER
    # ==========================================

    if optimizer_mode == "Based on Data (ML Optimizer)":

        st.caption(
            "Searches Rxn Temp, C2 Pressure, H2/C2, C4/C2, C6/C2, ICA and "
            "Al/Ti (with a fixed C2 feed rate) using the XGBoost + ANN "
            "ensemble to hit your target product quality."
        )

        feed_rate = st.number_input(

            "C2 Feed Rate (kg/h)",

            value=10000.0,

            key="opt_feed_rate"

        )

        col1,col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Target Product Properties"
            )

            target_mfi = st.number_input(

                "Target MFI @2.16 kg/cm²",

                value=5.0,

                key="opt_target_mfi"

            )

            target_productivity = st.number_input(

                "Target Productivity (kg PE/g cat)",

                value=7.0,

                key="opt_target_productivity"

            )

        with col2:

            st.markdown(
                "### &nbsp;"
            )

            target_density = st.number_input(

                "Target Density (g/cc)",

                value=0.935,

                format="%.4f",

                key="opt_target_density"

            )

        st.markdown("---")

        run_button = st.button(

            "Optimize Process",

            key="optimize_button"

        )

        if run_button:

            with st.spinner(
                "Running optimization..."
            ):

                result = optimize_process(

                    feed_rate,

                    target_mfi,

                    target_productivity,

                    target_density

                )

            best_temp = result.x[0]

            best_c2_pressure = result.x[1]

            best_h2_c2 = result.x[2]

            best_c4_c2 = result.x[3]

            best_c6_c2 = result.x[4]

            best_ica = result.x[5]

            best_al_ti = result.x[6]

            best_cat_rate = result.x[7]

            optimal_input = [

                best_temp,

                best_c2_pressure,

                best_h2_c2,

                best_c4_c2,

                best_c6_c2,

                best_ica,

                best_al_ti,

                feed_rate,

                best_cat_rate

            ]

            pred,std,preds = ensemble_predict(
                optimal_input
            )

            # Log this recommendation immediately, before the operator
            # has reacted to it, so it's on record either way — then
            # stash everything needed to redraw this result in
            # session_state. Without this, clicking the feedback widget
            # below (which reruns the whole script) would make the
            # result vanish, since it's only computed inside
            # `if run_button:`.

            model_setpoints = {

                "Rxn Temperature (°C)": float(best_temp),
                "C2 Pressure (kg/cm²)": float(best_c2_pressure),
                "H2/C2 ratio": float(best_h2_c2),
                "C4/C2 ratio": float(best_c4_c2),
                "C6/C2 ratio": float(best_c6_c2),
                "ICA (mol %)": float(best_ica),
                "Al/Ti ratio": float(best_al_ti),
                "Catalyst Rate (kg/h)": float(best_cat_rate)

            }

            predicted_quality = {

                "MFI @2.16 kg/cm2": float(pred[0]),
                "Productivity kg PE/g cat": float(pred[1]),
                "Density g/cc": float(pred[2])

            }

            predicted_std = {

                "MFI @2.16 kg/cm2": float(std[0]),
                "Productivity kg PE/g cat": float(std[1]),
                "Density g/cc": float(std[2])

            }

            rec_id = save_recommendation(

                mode="ML Optimizer",

                inputs={

                    "feed_rate": feed_rate,
                    "target_mfi": target_mfi,
                    "target_productivity": target_productivity,
                    "target_density": target_density

                },

                model_setpoints=model_setpoints,

                model_predicted_quality=predicted_quality,

                model_predicted_std=predicted_std

            )

            st.session_state["ml_opt_result"] = {

                "rec_id": rec_id,
                "model_setpoints": model_setpoints,
                "predicted_quality": predicted_quality,
                "confidence": float(
                    np.exp(-np.mean(std/np.abs(pred).clip(1e-6))) * 100
                )

            }

            # A fresh recommendation means any earlier "submitted"
            # banner no longer applies to *this* result.
            st.session_state.pop(
                f"ml_opt_submitted_{rec_id}", None
            )

        if "ml_opt_result" in st.session_state:

            res = st.session_state["ml_opt_result"]

            model_setpoints = res["model_setpoints"]
            predicted_quality = res["predicted_quality"]
            confidence = res["confidence"]
            rec_id = res["rec_id"]

            st.markdown("---")

            col1,col2 = st.columns(2)

            with col1:

                st.subheader(
                    "Recommended Setpoints"
                )

                st.metric(

                    "Rxn Temperature (°C)",

                    f"{model_setpoints['Rxn Temperature (°C)']:.2f}"

                )

                st.metric(

                    "C2 Pressure (kg/cm²)",

                    f"{model_setpoints['C2 Pressure (kg/cm²)']:.2f}"

                )

                st.metric(

                    "H2/C2 ratio",

                    f"{model_setpoints['H2/C2 ratio']:.3f}"

                )

                st.metric(

                    "C4/C2 ratio",

                    f"{model_setpoints['C4/C2 ratio']:.3f}"

                )

                st.metric(

                    "C6/C2 ratio",

                    f"{model_setpoints['C6/C2 ratio']:.3f}"

                )

                st.metric(

                    "ICA (mol %)",

                    f"{model_setpoints['ICA (mol %)']:.2f}"

                )

                st.metric(

                    "Al/Ti ratio",

                    f"{model_setpoints['Al/Ti ratio']:.1f}"

                )

                st.metric(

                    "Catalyst Rate (kg/h)",

                    f"{model_setpoints['Catalyst Rate (kg/h)']:.2f}"

                )

            with col2:

                st.subheader(
                    "Predicted Quality"
                )

                st.metric(

                    "Predicted MFI @2.16 kg/cm²",

                    f"{predicted_quality['MFI @2.16 kg/cm2']:.2f}"

                )

                st.metric(

                    "Predicted Productivity (kg PE/g cat)",

                    f"{predicted_quality['Productivity kg PE/g cat']:.2f}"

                )

                st.metric(

                    "Predicted Density (g/cc)",

                    f"{predicted_quality['Density g/cc']:.4f}"

                )

            st.markdown("---")

            comparison = pd.DataFrame({

                "Variable": list(model_setpoints.keys()),

                "Recommended": list(model_setpoints.values())

            })

            st.subheader(
                "Optimization Summary"
            )

            st.dataframe(

                comparison,

                use_container_width=True

            )

            st.subheader(
                "Model Confidence"
            )

            st.progress(
                int(min(confidence,100))
            )

            st.write(
                f"{confidence:.1f}%"
            )

            st.markdown("---")

            render_feedback_widget(
                rec_id,
                "ml_opt",
                model_setpoints
            )

    # ==========================================
    # MODE 2 — PATENT-BASED (STATISTICAL) GRADE TRANSITION
    # ==========================================

    else:

        st.caption(
            "Grade-transition setpoints following the methodology of "
            "published patents and research papers: to fix which direction the "
            "temperature and pressure setpoints move and the "
            "range they can move within. Everything the patent doesn't "
            "constrain — H2/C2, comonomer ratios, ICA and Al/Ti — is "
            "searched freely by the ML ensemble alongside them, with "
            "MFI and density weighted heavily and productivity allowed "
            "to be traded off."
        )

        col1,col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Product 1 (Outgoing Grade)"
            )

            p1_temp = st.number_input(

                "Product 1 Rxn Temperature (°C)",

                value=95.0,

                key="p1_temp"

            )

            p1_mi = st.number_input(

                "Product 1 Melt Index",

                value=5.0,

                key="p1_mi"

            )

            p1_pressure = st.number_input(

                "Product 1 Rate-Limiting Reactant Partial Pressure (psig)",

                value=280.0,

                key="p1_pressure"

            )

            p1_density = st.number_input(

                "Product 1 Density (g/cc)",

                value=0.935,

                format="%.4f",

                key="p1_density"

            )

        with col2:

            st.markdown(
                "### Product 2 (Desired Incoming Grade)"
            )

            p2_temp = st.number_input(

                "Product 2 Desired Rxn Temperature (°C)",

                value=90.0,

                key="p2_temp"

            )

            p2_mi = st.number_input(

                "Product 2 Desired Melt Index",

                value=8.0,

                key="p2_mi"

            )

            p2_density = st.number_input(

                "Product 2 Desired Density (g/cc)",

                value=0.930,

                format="%.4f",

                key="p2_density"

            )

        st.markdown("---")

        st.markdown(
            "### Search Settings"
        )

        col3,col4 = st.columns(2)

        with col3:

            search_feed_rate = st.number_input(

                "C2 Feed Rate (kg/h)",

                value=10000.0,

                key="patent_feed_rate"

            )

        with col4:

            soft_target_productivity = st.number_input(

                "Target Productivity (kg PE/g cat) — soft target",

                value=7.0,

                key="patent_target_productivity",

                help="Used only as a light tiebreaker in the search — "
                     "MFI and density are weighted 20x higher, so "
                     "productivity is free to come in below this if "
                     "that's what hitting spec requires."

            )

        st.markdown("---")

        st.markdown(
            "### Acceptance Criteria"
        )

        acceptable_tol = st.slider(

            "Acceptable range around target MI / Density (%)",

            min_value=1,

            max_value=20,

            value=5,

            key="acceptable_tol",

            help="How close average MI and density need to settle to "
                 "the Product 2 targets before the transition counts "
                 "as complete (Step 5)."

        )

        st.markdown("---")

        calc_button = st.button(

            "Search Transition Setpoints",

            key="patent_calc_button"

        )

        if calc_button:

            with st.spinner(

                "Searching temperature, pressure and free process levers "
                "within their allowed ranges..."

            ):

                setpoints = patent_grade_transition_setpoints(

                    p1_temp,

                    p1_mi,

                    p1_pressure,

                    p2_temp,

                    p2_mi,

                    p2_density,

                    search_feed_rate,

                    soft_target_productivity

                )

            # Log the recommendation and stash everything needed to
            # redraw it, same reasoning as Mode 1: the feedback widget
            # below reruns the whole script, and without session_state
            # this whole result would disappear the instant the operator
            # touches it.

            model_setpoints = {

                "Refined Rxn Temp Setpoint (°C)": float(setpoints["temp_setpoint_refined"]),
                "Reactant Pressure Setpoint (psig)": float(setpoints["pressure_setpoint"]),
                "Melt Index Setpoint": float(setpoints["mi_setpoint"]),
                "H2/C2": float(setpoints["h2_c2"]),
                "C4/C2": float(setpoints["c4_c2"]),
                "C6/C2": float(setpoints["c6_c2"]),
                "ICA (mol %)": float(setpoints["ica"]),
                "Al/Ti": float(setpoints["al_ti"]),
                "Catalyst Rate (kg/h)": float(setpoints["cat_rate"])

            }

            predicted_quality = {

                "MFI @2.16 kg/cm2": float(setpoints["predicted_mfi"]),
                "Productivity kg PE/g cat": float(setpoints["predicted_productivity"]),
                "Density g/cc": float(setpoints["predicted_density"])

            }

            rec_id = save_recommendation(

                mode="Patent + Knowledge Base",

                inputs={

                    "p1_temp": p1_temp,
                    "p1_mi": p1_mi,
                    "p1_pressure": p1_pressure,
                    "p1_density": p1_density,
                    "p2_temp": p2_temp,
                    "p2_mi": p2_mi,
                    "p2_density": p2_density,
                    "search_feed_rate": search_feed_rate,
                    "soft_target_productivity": soft_target_productivity,
                    "acceptable_tol": acceptable_tol

                },

                model_setpoints=model_setpoints,

                model_predicted_quality=predicted_quality

            )

            st.session_state["patent_result"] = {

                "rec_id": rec_id,
                "setpoints": setpoints,
                "model_setpoints": model_setpoints,
                "p2_mi": p2_mi,
                "p2_density": p2_density,
                "acceptable_tol": acceptable_tol

            }

            st.session_state.pop(
                f"patent_submitted_{rec_id}", None
            )

        if "patent_result" in st.session_state:

            pres = st.session_state["patent_result"]

            rec_id = pres["rec_id"]
            setpoints = pres["setpoints"]
            model_setpoints = pres["model_setpoints"]
            p2_mi = pres["p2_mi"]
            p2_density = pres["p2_density"]
            acceptable_tol = pres["acceptable_tol"]

            st.markdown("---")

            direction_note = (
                "Product 2 MI is higher than Product 1 MI, reading patent US5627242A..." if setpoints["mi_increasing"]
                else "Product 2 MI is lower than Product 1 MI (or equal), reading patent US5627242A... "
            )

            st.info(f"Direction rule in effect: **{direction_note}**")

            st.markdown("### Step 1 — Initial Temperature Setpoint")

            st.caption(
                "Drop immediately to Product 2's temperature only if it's "
                "lower than Product 1's; otherwise hold at Product 1's "
                "temperature until Step 3 refines it."
            )

            st.metric(

                "Initial Rxn Temp Setpoint (°C)",

                f"{setpoints['temp_setpoint_initial']:.2f}"

            )

            st.markdown("### Step 2 — Melt Index Setpoint")

            st.caption(
                "Legal range is 0-150% higher (if MI is increasing) or "
                "0-70% lower (if decreasing) than the Product 2 target. "
                "Set to whatever MI the model predicts Steps 3-4 will "
                "actually produce, clipped into that legal range."
            )

            st.metric(

                "Melt Index Setpoint",

                f"{setpoints['mi_setpoint']:.2f}",

                delta=f"{setpoints['mi_pct']:+.0f}% vs Product 2 target"

            )

            col1,col2 = st.columns(2)

            with col1:

                st.markdown("### Step 3 — Refined Temperature Setpoint")

                st.caption(
                    "Searched within 1-15°C above Product 2's "
                    "temperature (MI increasing) or below (MI "
                    "decreasing) for the value closest to target quality."
                )

                st.metric(

                    "Refined Rxn Temp Setpoint (°C)",

                    f"{setpoints['temp_setpoint_refined']:.2f}",

                    delta=f"{setpoints['temp_delta']:.1f}°C trim (of 1-15°C range)"

                )

            with col2:

                st.markdown("### Step 4 — Reactant Pressure Setpoint")

                st.caption(
                    "Searched within 1-25 psig below Product 1's "
                    "pressure (MI increasing) or above (MI decreasing) "
                    "for the value closest to target quality."
                )

                st.metric(

                    "Rate-Limiting Reactant Pressure Setpoint (psig)",

                    f"{setpoints['pressure_setpoint']:.2f}",

                    delta=f"{setpoints['pressure_delta']:.1f} psig trim (of 1-25 psig range)"

                )

            st.markdown("---")

            st.markdown("### Supporting Levers (Not Constrained by the Patent)")

            st.caption(

                "The patent's rules only cover temperature, MI and "
                "reactant pressure. These are the H2/C2, comonomer "
                "ratios, ICA and Al/Ti values the search found — "
                "searched jointly with Steps 3-4 above, weighting MFI "
                "and density heavily and letting productivity flex."

            )

            lc1,lc2,lc3,lc4,lc5 = st.columns(5)

            with lc1:

                st.metric("H2/C2", f"{setpoints['h2_c2']:.3f}")

            with lc2:

                st.metric("C4/C2", f"{setpoints['c4_c2']:.3f}")

            with lc3:

                st.metric("C6/C2", f"{setpoints['c6_c2']:.3f}")

            with lc4:

                st.metric("ICA (mol %)", f"{setpoints['ica']:.2f}")

            with lc5:

                st.metric("Al/Ti", f"{setpoints['al_ti']:.1f}")

            st.metric(

                "Catalyst Rate (kg/h)",

                f"{setpoints['cat_rate']:.2f}"

            )

            st.markdown("---")

            st.subheader(
                "Step 5 — Maintain Until Within Acceptable Range"
            )

            mi_low = p2_mi * (1 - acceptable_tol / 100.0)
            mi_high = p2_mi * (1 + acceptable_tol / 100.0)
            density_low = p2_density * (1 - acceptable_tol / 100.0)
            density_high = p2_density * (1 + acceptable_tol / 100.0)

            st.write(

                f"Hold the setpoints above until the polymerization "
                f"product's **average melt index** settles between "
                f"**{mi_low:.2f} and {mi_high:.2f}**, and its **average "
                f"density** settles between **{density_low:.4f} and "
                f"{density_high:.4f} g/cc** (±{acceptable_tol}% around "
                f"the Product 2 targets)."

            )

            # ==========================================
            # WHAT THE SEARCH ACTUALLY EXPECTS TO HAPPEN
            # ==========================================
            # This is the same ML ensemble the search above already used
            # to score every candidate (temp_delta, pressure_delta,
            # H2/C2, C4/C2, C6/C2, ICA, Al/Ti, cat_rate) combination —
            # we're just surfacing its prediction for the winning
            # combination, rather than re-running it separately.

            st.markdown("---")

            st.markdown("### Predicted Outcome of the Full Setpoint Combination")

            st.caption(

                "What the ML ensemble predicts the searched combination "
                "above (Steps 3-4 plus the supporting levers) will "
                "actually produce — this is the same model the search "
                "optimized against."

            )

            cc1,cc2,cc3 = st.columns(3)

            with cc1:

                st.metric(

                    "Predicted MFI @2.16 kg/cm²",

                    f"{setpoints['predicted_mfi']:.2f}",

                    delta=f"target {p2_mi:.2f}"

                )

            with cc2:

                st.metric(

                    "Predicted Productivity (kg PE/g cat)",

                    f"{setpoints['predicted_productivity']:.2f}"

                )

            with cc3:

                st.metric(

                    "Predicted Density (g/cc)",

                    f"{setpoints['predicted_density']:.4f}",

                    delta=f"target {p2_density:.4f}"

                )

            mi_in_range = mi_low <= setpoints["predicted_mfi"] <= mi_high
            density_in_range = density_low <= setpoints["predicted_density"] <= density_high

            if mi_in_range and density_in_range:

                st.success(

                    "Predicted MI and density both fall inside the Step "
                    "5 acceptable range for Product 2 — this search "
                    "found a setpoint combination consistent with a "
                    "completed transition."

                )

            else:

                off_target = []
                if not mi_in_range:
                    off_target.append("MI")
                if not density_in_range:
                    off_target.append("density")

                st.warning(

                    f"Even the best setpoints found within the patent's "
                    f"legal ranges still leave predicted {' and '.join(off_target)} "
                    f"outside the Step 5 acceptable range — the H2/C2, "
                    f"comonomer ratios, ICA or Al/Ti (which the patent "
                    f"doesn't constrain) likely need adjusting too, e.g. "
                    f"with the data-driven optimizer mode."

                )

            st.markdown("---")

            render_feedback_widget(
                rec_id,
                "patent",
                model_setpoints
            )



# =====================================================
# MODEL INSIGHTS
# =====================================================



with tab4:

    st.subheader(
        "Model Agreement"
    )

    sample = df.iloc[0]

    sample_input = [

        sample["Rxn Temp. oC"],
        sample["C2 Pressure kg/cm2"],
        sample["H2/C2"],
        sample["C4/C2"],
        sample["C6/C2"],
        sample["ICA mol %"],
        sample["Al/Ti"],
        sample["Feed rate (kg/h)"],
        sample["Catalyst rate (kg/h)"]

    ]

    pred,std,preds = ensemble_predict(
        sample_input
    )

    agreement = pd.DataFrame({

        "Model":[


            "XGBoost",

            "ANN"

        ],

        "MFI":[

            preds[0][0],
            preds[1][0],
         

        ],

        "Productivity":[

            preds[0][1],
            preds[1][1],
          

        ],

        "Density":[

            preds[0][2],
            preds[1][2]

        ]

    })

    st.dataframe(
        agreement,
        use_container_width=True
    )

    fig1 = px.bar(

        agreement,

        x="Model",

        y="MFI",

        title="MFI Prediction Comparison"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.bar(

        agreement,

        x="Model",

        y="Productivity",

        title="Productivity Prediction Comparison"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.bar(

        agreement,

        x="Model",

        y="Density",

        title="Density Prediction Comparison"

    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =====================================================
# ANOMALY DETECTION (time-series aware)
# =====================================================

with tab5:

    st.subheader(
        "Time-Series Anomaly Detection"
    )

    st.caption(
        "Rolling-window Isolation Forest — each point is scored using the "
        "mean and standard deviation of every variable over a sliding "
        "window, so drifts and pattern shifts are caught, not just "
        "single-row spikes."
    )

    col1,col2 = st.columns(2)

    with col1:

        window = st.slider(

            "Rolling window size (samples)",

            min_value=5,

            max_value=100,

            value=20,

            step=5,

            key="anomaly_window"

        )

    with col2:

        contamination = st.slider(

            "Expected anomaly rate",

            min_value=0.01,

            max_value=0.10,

            value=0.03,

            step=0.01,

            key="anomaly_contamination"

        )

    results = run_ts_anomaly_detection(

        df,

        window=window,

        contamination=contamination

    )

    total_points = len(results)

    total_anomalies = int(results["is_anomaly"].sum())

    anomaly_rate = 100 * total_anomalies / total_points if total_points else 0

    latest = results.iloc[-1]

    st.markdown("---")

    m1,m2,m3,m4 = st.columns(4)

    with m1:

        st.metric(

            "Points analyzed",

            f"{total_points}"

        )

    with m2:

        st.metric(

            "Anomalies detected",

            f"{total_anomalies}"

        )

    with m3:

        st.metric(

            "Anomaly rate",

            f"{anomaly_rate:.1f}%"

        )

    with m4:

        latest_status = "🔴 Anomaly" if latest["is_anomaly"] else "🟢 Normal"

        st.metric(

            "Latest reading",

            latest_status,

            delta=f"Health {latest['health']:.1f}"

        )

    st.markdown("---")

    st.markdown("### Health Score Over Time")

    fig_health = px.line(

        results,

        x=results.index,

        y="health",

        title="Process Health Score (rolling Isolation Forest)"

    )

    fig_health.update_layout(

        xaxis_title="Sample Number",

        yaxis_title="Health Score"

    )

    anomaly_points = results[results["is_anomaly"]]

    fig_health.add_scatter(

        x=anomaly_points.index,

        y=anomaly_points["health"],

        mode="markers",

        marker=dict(color="red", size=8, symbol="x"),

        name="Anomaly"

    )

    st.plotly_chart(

        fig_health,

        use_container_width=True

    )

    st.markdown("### Variable Trend with Anomaly Overlay")

    selected_var = st.selectbox(

        "Select Variable",

        feature_names + target_names,

        key="anomaly_var"

    )

    fig_var = px.line(

        results,

        x=results.index,

        y=selected_var,

        title=f"{selected_var} with Detected Anomalies"

    )

    fig_var.add_scatter(

        x=anomaly_points.index,

        y=anomaly_points[selected_var],

        mode="markers",

        marker=dict(color="red", size=8, symbol="x"),

        name="Anomaly"

    )

    fig_var.update_layout(

        xaxis_title="Sample Number",

        yaxis_title=selected_var

    )

    st.plotly_chart(

        fig_var,

        use_container_width=True

    )

    st.markdown("### Detected Anomaly Events")

    if total_anomalies == 0:

        st.info(

            "No anomalies detected at the current window size / "
            "contamination setting."

        )

    else:

        # Only request columns that actually exist — avoids a KeyError
        # if the underlying data file ever has different column names.
        wanted_cols = feature_names + target_names + ["anomaly_score","health"]

        display_cols = [c for c in wanted_cols if c in anomaly_points.columns]

        anomaly_table = anomaly_points[display_cols].sort_values(

            "anomaly_score"

        )

        st.dataframe(

            anomaly_table,

            use_container_width=True

        )

        # ==========================================
        # WHY WAS THIS FLAGGED? (dependency-based explanation)
        # ==========================================

        st.markdown("### Why Was This Flagged?")

        st.caption(

            "For a selected anomaly, this shows how far each variable "
            "sat from its own rolling average (in standard-deviation "
            "units) at that moment, plus which other variables it's "
            "historically correlated with — the combination points at "
            "*which* relationship broke down, not just *that* something did."

        )

        chosen_idx = st.selectbox(

            "Select an anomaly (by sample index)",

            anomaly_table.index.tolist(),

            key="anomaly_explain_idx"

        )

        zscore_cols = [f"{c}_zscore" for c in feature_names]

        z_row = results.loc[chosen_idx, zscore_cols]

        z_row.index = feature_names

        z_row = z_row.sort_values(key=lambda s: s.abs(), ascending=False)

        fig_z = px.bar(

            x=z_row.index,

            y=z_row.values,

            title=f"Variable Deviation at Sample {chosen_idx} (std units)",

            labels={"x":"Variable","y":"Deviation (rolling z-score)"}

        )

        fig_z.add_hline(y=2, line_dash="dot", line_color="red")
        fig_z.add_hline(y=-2, line_dash="dot", line_color="red")

        st.plotly_chart(

            fig_z,

            use_container_width=True

        )

        top_var = z_row.index[0]

        corr_matrix = get_feature_correlations(df, feature_names)

        related = corr_matrix[top_var].drop(top_var).sort_values(

            key=lambda s: s.abs(),

            ascending=False

        )

        second_var = related.index[0]

        second_corr = related.iloc[0]

        relationship = "positively" if second_corr > 0 else "negatively"

        st.info(

            f"**{top_var}** deviated the most from its rolling average "
            f"(z = {z_row.iloc[0]:.2f}). Historically it is {relationship} "
            f"correlated with **{second_var}** (r = {second_corr:.2f}) — "
            f"worth checking whether that pair moved together as expected "
            f"or decoupled at this point, since that's usually what "
            f"separates a process drift from a sensor glitch."

        )

# =====================================================
# OPERATOR FEEDBACK & MODEL REFINEMENT
# =====================================================

with tab6:

    st.subheader(
        "Operator Feedback & Model Refinement"
    )

    st.caption(
        "Every setpoint recommendation from the Optimizer tab is logged "
        "here the moment it's generated. Accept / modify / reject "
        "decisions — and the reasons behind them — are what let this "
        "model be refined against real plant experience over time, "
        "rather than historical data alone."
    )

    feedback_log = load_feedback_log()

    if feedback_log.empty:

        st.info(
            "No recommendations logged yet. Generate one from the "
            "Optimizer tab and give it feedback to see it here."
        )

    else:

        total = len(feedback_log)

        decision_counts = feedback_log["operator_decision"].value_counts()

        n_pending = int(decision_counts.get("pending", 0))
        n_accepted = int(decision_counts.get("accepted", 0))
        n_modified = int(decision_counts.get("modified", 0))
        n_rejected = int(decision_counts.get("rejected", 0))
        n_reviewed = total - n_pending

        st.markdown("### Summary")

        m1,m2,m3,m4,m5 = st.columns(5)

        with m1:

            st.metric("Total recommendations", f"{total}")

        with m2:

            st.metric("Awaiting review", f"{n_pending}")

        with m3:

            accept_rate = 100 * n_accepted / n_reviewed if n_reviewed else 0

            st.metric(
                "Accepted as-is",
                f"{n_accepted}",
                delta=f"{accept_rate:.0f}% of reviewed"
            )

        with m4:

            modify_rate = 100 * n_modified / n_reviewed if n_reviewed else 0

            st.metric(
                "Modified",
                f"{n_modified}",
                delta=f"{modify_rate:.0f}% of reviewed"
            )

        with m5:

            reject_rate = 100 * n_rejected / n_reviewed if n_reviewed else 0

            st.metric(
                "Rejected",
                f"{n_rejected}",
                delta=f"{reject_rate:.0f}% of reviewed"
            )

        st.markdown("---")

        st.markdown("### Decisions Over Time")

        decision_df = feedback_log.copy()

        decision_df["timestamp"] = pd.to_datetime(
            decision_df["timestamp"], errors="coerce"
        )

        decision_df = decision_df.dropna(subset=["timestamp"])

        if decision_df.empty:

            st.info("Not enough timestamped data yet to plot a trend.")

        else:

            fig_decisions = px.histogram(

                decision_df,

                x="timestamp",

                color="operator_decision",

                title="Recommendations by Decision, Over Time"

            )

            st.plotly_chart(
                fig_decisions,
                use_container_width=True
            )

        st.markdown("---")

        st.markdown("### Where Operators Override the Model")

        st.caption(
            "For every 'modified' recommendation, this shows how far "
            "the operator's setpoint sat from the model's, averaged "
            "per variable. A consistent, sizeable average delta on a "
            "variable is a systematic bias worth correcting before the "
            "next retraining pass — before touching the model at all, "
            "this alone tells you where it's directionally off."
        )

        modified_rows = feedback_log[
            feedback_log["operator_decision"] == "modified"
        ]

        if modified_rows.empty:

            st.info("No modified recommendations yet.")

        else:

            deltas = {}

            for _, row in modified_rows.iterrows():

                try:

                    model_vals = json.loads(row["model_setpoints"])
                    operator_vals = json.loads(row["operator_setpoints"])

                except (TypeError, ValueError):

                    continue

                for label, op_val in operator_vals.items():

                    model_val = model_vals.get(label)

                    if model_val is None:
                        continue

                    deltas.setdefault(label, []).append(op_val - model_val)

            if not deltas:

                st.info(
                    "No comparable setpoint values found in the "
                    "modified rows yet."
                )

            else:

                delta_summary = pd.DataFrame({

                    "Variable": list(deltas.keys()),
                    "Avg Operator - Model Delta": [
                        float(np.mean(v)) for v in deltas.values()
                    ],
                    "N": [len(v) for v in deltas.values()]

                })

                fig_delta = px.bar(

                    delta_summary,

                    x="Variable",

                    y="Avg Operator - Model Delta",

                    title="Average Operator Override vs Model Recommendation"

                )

                st.plotly_chart(
                    fig_delta,
                    use_container_width=True
                )

                st.dataframe(
                    delta_summary,
                    use_container_width=True
                )

        st.markdown("---")

        st.markdown("### Full Feedback Log")

        display_cols = [

            "id",
            "timestamp",
            "mode",
            "operator_decision",
            "operator_reason",
            "feedback_timestamp"

        ]

        st.dataframe(

            feedback_log[display_cols].sort_values(
                "timestamp", ascending=False
            ),

            use_container_width=True

        )

        with st.expander("Inspect a single recommendation in full"):

            chosen_id = st.selectbox(

                "Recommendation ID",

                feedback_log["id"].tolist(),

                key="feedback_inspect_id"

            )

            chosen_row = feedback_log[
                feedback_log["id"] == chosen_id
            ].iloc[0]

            insp1,insp2 = st.columns(2)

            with insp1:

                st.markdown("**Model recommended**")

                st.json(
                    json.loads(chosen_row["model_setpoints"])
                    if chosen_row["model_setpoints"] else {}
                )

                st.markdown("**Model predicted quality**")

                st.json(
                    json.loads(chosen_row["model_predicted_quality"])
                    if chosen_row["model_predicted_quality"] else {}
                )

            with insp2:

                st.markdown("**Operator decision**")

                st.write(chosen_row["operator_decision"])

                st.markdown("**Operator setpoints (if modified)**")

                st.json(
                    json.loads(chosen_row["operator_setpoints"])
                    if chosen_row["operator_setpoints"] else {}
                )

                st.markdown("**Operator reason**")

                st.write(chosen_row["operator_reason"] or "—")

                if chosen_row["actual_quality"]:

                    st.markdown("**Actual plant outcome**")

                    st.json(json.loads(chosen_row["actual_quality"]))

        st.markdown("---")

        st.markdown("### Record Actual Plant Outcome")

        st.caption(
            "Once a recommendation has actually run on the plant and "
            "the real product quality is known, record it here. This "
            "is what turns a row from an opinion into a genuine "
            "(input, setpoint used, outcome) example for the next "
            "retraining pass."
        )

        outcome_candidates = feedback_log[
            feedback_log["operator_decision"] != "pending"
        ]["id"].tolist()

        if not outcome_candidates:

            st.info(
                "No reviewed recommendations available yet to attach "
                "an outcome to."
            )

        else:

            outcome_id = st.selectbox(

                "Recommendation ID",

                outcome_candidates,

                key="outcome_rec_id"

            )

            oc1,oc2,oc3 = st.columns(3)

            with oc1:

                actual_mfi = st.number_input(

                    "Actual MFI @2.16 kg/cm²",

                    value=0.0,

                    key="actual_mfi",

                    format="%.3f"

                )

            with oc2:

                actual_prod = st.number_input(

                    "Actual Productivity (kg PE/g cat)",

                    value=0.0,

                    key="actual_prod",

                    format="%.3f"

                )

            with oc3:

                actual_density = st.number_input(

                    "Actual Density (g/cc)",

                    value=0.0,

                    key="actual_density",

                    format="%.4f"

                )

            if st.button("Save Actual Outcome", key="save_outcome_button"):

                record_actual_outcome(

                    outcome_id,

                    {
                        "MFI @2.16 kg/cm2": actual_mfi,
                        "Productivity kg PE/g cat": actual_prod,
                        "Density g/cc": actual_density
                    }

                )

                st.success(f"Actual outcome recorded for `{outcome_id}`.")

st.markdown("---")

st.caption(
    "Advanced Process Control Dashboard | Ensemble ML Optimizer"
)
