
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

# =====================================================
# OPTIMIZER
# =====================================================

def optimize_process(

    feed_rate,
    c2_pressure,
    target_mfi,
    target_productivity,
    target_density

):

    def objective(x):

        rxn_temp = x[0]
        h2_c2 = x[1]
        c4_c2 = x[2]
        c6_c2 = x[3]
        ica = x[4]
        al_ti = x[5]
        cat_rate = x[6]

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
tab1,tab2,tab3,tab4,tab5 = st.tabs(

    [

        "Live PFD",

        "Historical Data",

        "Optimizer",

        "Model diagnostics",

        "Anomaly Detection"


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
top:294px;
width:0;height:0;
border-top:7px solid transparent;
border-bottom:7px solid transparent;
border-left:14px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:45px;
top:240px;
color:#0D47A1;
font-size:14px;
">
<b>C2 FEED</b><br>
Flow = {st.session_state.feed_rate:.0f} kg/h<br>
P = {st.session_state.c2_pressure:.1f} kg/cm&sup2;
</div>

<!-- ================= COMONOMER (C4/C2, C6/C2 ratios) ================= -->
<!-- horizontal line from x=140 to x=420 (reactor left edge) -->

<div style="
position:absolute;
left:140px;
top:380px;
width:280px;
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
left:145px;
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
left:462px;
top:96px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-top:14px solid #1976D2;
"></div>

<div style="
position:absolute;
left:400px;
top:10px;
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
left:542px;
top:96px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-top:14px solid #8E24AA;
"></div>

<div style="
position:absolute;
left:530px;
top:10px;
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
background:white;
">

<!-- Disengaging Zone -->

<div style="
height:90px;
background:#E3F2FD;
border-bottom:3px dashed #1976D2;
text-align:center;
padding-top:15px;
font-weight:bold;
color:#0D47A1;
">
DISENGAGING<br>ZONE
</div>

<!-- Fluidized Bed -->

<div style="
height:270px;
background:
radial-gradient(circle at 20% 90%, white 5px, transparent 6px),
radial-gradient(circle at 40% 75%, white 6px, transparent 7px),
radial-gradient(circle at 70% 82%, white 5px, transparent 6px),
radial-gradient(circle at 60% 60%, white 7px, transparent 8px),
linear-gradient(to top,#4CAF50,#81C784,#A5D6A7);
text-align:center;
padding-top:25px;
color:white;
font-weight:bold;
font-size:14px;
">

FLUIDIZED BED

<br><br>

Rxn Temperature<br>
{st.session_state.rxn_temp:.1f} &deg;C

<br><br>

C2 Pressure<br>
{st.session_state.c2_pressure:.1f} kg/cm&sup2;

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
top:460px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-bottom:14px solid red;
"></div>

<div style="
position:absolute;
left:395px;
top:545px;
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
top:460px;
width:0;height:0;
border-left:7px solid transparent;
border-right:7px solid transparent;
border-bottom:14px solid orange;
"></div>

<div style="
position:absolute;
left:525px;
top:545px;
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
left:818px;
top:284px;
width:0;height:0;
border-top:7px solid transparent;
border-bottom:7px solid transparent;
border-left:14px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:640px;
top:195px;
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

        if add_trendline:

            fig = px.scatter(

                df,

                x=x_var,

                y=y_var,

                trendline="ols",

                title=f"{x_var} vs {y_var}"

            )

        else:

            fig = px.scatter(

                df,

                x=x_var,

                y=y_var,

                title=f"{x_var} vs {y_var}"

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

    col1,col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Current Plant Conditions"
        )

        feed_rate = st.number_input(

            "C2 Feed Rate (kg/h)",

            value=10000.0,

            key="opt_feed_rate"

        )

        c2_pressure = st.number_input(

            "C2 Pressure (kg/cm²)",

            value=20.0,

            key="opt_c2_pressure"

        )

    with col2:

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

                c2_pressure,

                target_mfi,

                target_productivity,

                target_density

            )

        best_temp = result.x[0]

        best_h2_c2 = result.x[1]

        best_c4_c2 = result.x[2]

        best_c6_c2 = result.x[3]

        best_ica = result.x[4]

        best_al_ti = result.x[5]

        best_cat_rate = result.x[6]

        optimal_input = [

            best_temp,

            c2_pressure,

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

        st.markdown("---")

        col1,col2 = st.columns(2)

        with col1:

            st.subheader(
                "Recommended Setpoints"
            )

            st.metric(

                "Rxn Temperature (°C)",

                f"{best_temp:.2f}"

            )

            st.metric(

                "H2/C2 ratio",

                f"{best_h2_c2:.3f}"

            )

            st.metric(

                "C4/C2 ratio",

                f"{best_c4_c2:.3f}"

            )

            st.metric(

                "C6/C2 ratio",

                f"{best_c6_c2:.3f}"

            )

            st.metric(

                "ICA (mol %)",

                f"{best_ica:.2f}"

            )

            st.metric(

                "Al/Ti ratio",

                f"{best_al_ti:.1f}"

            )

            st.metric(

                "Catalyst Rate (kg/h)",

                f"{best_cat_rate:.2f}"

            )

        with col2:

            st.subheader(
                "Predicted Quality"
            )

            st.metric(

                "Predicted MFI @2.16 kg/cm²",

                f"{pred[0]:.2f}"

            )

            st.metric(

                "Predicted Productivity (kg PE/g cat)",

                f"{pred[1]:.2f}"

            )

            st.metric(

                "Predicted Density (g/cc)",

                f"{pred[2]:.4f}"

            )

        st.markdown("---")

        comparison = pd.DataFrame({

            "Variable":[

                "Rxn Temperature (°C)",

                "H2/C2",

                "C4/C2",

                "C6/C2",

                "ICA (mol %)",

                "Al/Ti",

                "Catalyst Rate (kg/h)"

            ],

            "Recommended":[

                best_temp,

                best_h2_c2,

                best_c4_c2,

                best_c6_c2,

                best_ica,

                best_al_ti,

                best_cat_rate

            ]

        })

        st.subheader(
            "Optimization Summary"
        )

        st.dataframe(

            comparison,

            use_container_width=True

        )

        confidence = np.exp(
            -np.mean(std/np.abs(pred).clip(1e-6))
        ) * 100

        st.subheader(
            "Model Confidence"
        )

        st.progress(
            int(min(confidence,100))
        )

        st.write(
            f"{confidence:.1f}%"
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

st.markdown("---")

st.caption(
    "Advanced Process Control Dashboard | Ensemble ML Optimizer"
)
