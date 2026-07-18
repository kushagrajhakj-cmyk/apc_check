
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

            nn.Linear(6,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,2)

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

    "Reactor_Pressure",
    "Feed_Temperature",
    "Feed_Rate",
    "Reactor_Temperature",
    "Hydrogen_Flow",
    "Catalyst_Loading"

]

# Product-quality columns aren't guaranteed to exist under these exact
# names in every data file, so detect them rather than hardcoding —
# this is also what caused the earlier KeyError.
target_names = [c for c in ["MFI", "Yield"] if c in df.columns]


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

    pressure,
    feed_temp,
    feed_rate,
    target_mfi,
    target_yield

):

    def objective(x):

        reactor_temp = x[0]
        hydrogen = x[1]
        catalyst = x[2]

        input_vector = [

            pressure,
            feed_temp,
            feed_rate,

            reactor_temp,
            hydrogen,
            catalyst

        ]

        pred,std,_ = ensemble_predict(
            input_vector
        )

        loss = (

            (pred[0]-target_mfi)**2

            +

            (pred[1]-target_yield)**2

            +

            0.5*np.sum(std)

        )

        return loss

    bounds = [

        (200,260),
        (10,60),
        (0.5,3.0)

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
        "feed_temp": float(current["Feed_Temperature"]),
        "feed_rate": float(current["Feed_Rate"]),
        "reactor_temp": float(current["Reactor_Temperature"]),
        "reactor_pressure": float(current["Reactor_Pressure"]),
        "hydrogen_flow": float(current["Hydrogen_Flow"]),
        "catalyst_loading": float(current["Catalyst_Loading"]),
        "mfi_pred": float(current["MFI"]),
        "yield_pred": float(current["Yield"])
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

<!-- ================= FEED ================= -->

<div style="
position:absolute;
left:40px;
top:270px;
width:260px;
border-top:4px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:45px;
top:205px;
color:#0D47A1;
font-size:14px;
">
<b>FEED</b><br>
T = {st.session_state.feed_temp:.1f} °C<br>
Flow = {st.session_state.feed_rate:.1f} m³/h
</div>

<!-- ================= COMONOMER ================= -->

<div style="
position:absolute;
left:180px;
top:185px;
width:120px;
border-top:3px solid green;
"></div>

<div style="
position:absolute;
left:70px;
top:140px;
color:green;
font-size:13px;
">
<b>COMONOMER</b><br>
Flow = 15 kg/h<br>
T = 55 °C
</div>

<!-- ================= H2 ================= -->

<div style="
position:absolute;
left:510px;
top:25px;
width:4px;
height:90px;
background:#1976D2;
"></div>

<div style="
position:absolute;
left:435px;
top:5px;
color:#1565C0;
text-align:center;
font-size:14px;
">
<b>H₂</b><br>
Flow = {st.session_state.hydrogen_flow:.1f} kg/h<br>
T = 45 °C
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

Temperature<br>
{st.session_state.reactor_temp:.1f} °C

<br><br>

Pressure<br>
{st.session_state.reactor_pressure:.1f} bar

<br><br>

ICA = 8 wt%

</div>

</div>

<!-- ================= CATALYST ================= -->

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
left:350px;
top:540px;
color:red;
text-align:center;
font-size:13px;
">
<b>CATALYST</b><br>
{st.session_state.catalyst_loading:.2f} kg/h
</div>

<!-- ================= COCATALYST ================= -->

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
left:560px;
top:540px;
color:orange;
text-align:center;
font-size:13px;
">
<b>COCATALYST</b><br>
5.0 kg/h
</div>

<!-- ================= PRODUCT ================= -->

<div style="
position:absolute;
left:600px;
top:290px;
width:220px;
border-top:4px solid #1E88E5;
"></div>

<div style="
position:absolute;
left:835px;
top:225px;
color:#0D47A1;
font-size:14px;
">
<b>PRODUCT</b><br>

Productivity = {st.session_state.yield_pred:.2f} t/h<br>

MFI = {st.session_state.mfi_pred:.2f}<br>

Density = 0.918 g/cm³

</div>

</div>
""",
unsafe_allow_html=True
)


    st.markdown("### Process Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.number_input(
         "Feed Temperature (°C)",
          key="feed_temp"
         )

        st.number_input(
         "Feed Rate",
         key="feed_rate"
        )

    with c2:

        st.number_input(
            "Reactor Temperature (°C)",
            key="reactor_temp"
        )

        st.number_input(
            "Reactor Pressure (bar)",
            key="reactor_pressure"
        )

    with c3:

        st.number_input(
            "Hydrogen Flow",
            key="hydrogen_flow"
        )

        st.number_input(
            "Catalyst Loading",
            key="catalyst_loading"
        )



    predict_button = st.button(
        "Predict Product Quality",
         use_container_width=True
        )

    if predict_button:

        input_vector = [

            st.session_state.reactor_pressure,
            st.session_state.feed_temp,
            st.session_state.feed_rate,
            st.session_state.reactor_temp,
            st.session_state.hydrogen_flow,
            st.session_state.catalyst_loading

        ]

        pred, std, preds = ensemble_predict(
            input_vector
        )

        st.session_state.mfi_pred = float(pred[0])
        st.session_state.yield_pred = float(pred[1])

        confidence = np.exp(
            -np.mean(std)
        ) * 100

        st.success(
            f"Prediction Completed | Confidence = {confidence:.1f}%"
        )

        mfi = st.session_state.mfi_pred
        yield_value = st.session_state.yield_pred

        




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

        pressure = st.number_input(

            "Reactor Pressure",

            value=22.0,

            key="opt_pressure"

        )

        feed_temp = st.number_input(

            "Feed Temperature",

            value=75.0,

            key="opt_feed_temp"

        )

        feed_rate = st.number_input(

            "Feed Rate",

            value=120.0,

            key="opt_feed_rate"

        )

    with col2:

        st.markdown(
            "### Target Product Properties"
        )

        target_mfi = st.number_input(

            "Target MFI",

            value=35.0,

            key="opt_target_mfi"

        )

        target_yield = st.number_input(

            "Target yield",

            value=90.0,

            key="opt_target_yield"

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

                pressure,

                feed_temp,

                feed_rate,

                target_mfi,

                target_yield

            )

        best_temp = result.x[0]

        best_h2 = result.x[1]

        best_cat = result.x[2]

        optimal_input = [

            pressure,

            feed_temp,

            feed_rate,

            best_temp,

            best_h2,

            best_cat

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

                "Reactor Temperature",

                f"{best_temp:.2f}"

            )

            st.metric(

                "Hydrogen Flow",

                f"{best_h2:.2f}"

            )

            st.metric(

                "Catalyst Loading",

                f"{best_cat:.2f}"

            )

        with col2:

            st.subheader(
                "Predicted Quality"
            )

            st.metric(

                "Predicted MFI",

                f"{pred[0]:.2f}"

            )

            st.metric(

                "Predicted yield",

                f"{pred[1]:.2f}"

            )

        st.markdown("---")

        comparison = pd.DataFrame({

            "Variable":[

                "Reactor Temperature",

                "Hydrogen Flow",

                "Catalyst Loading"

            ],

            "Recommended":[

                best_temp,

                best_h2,

                best_cat

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
            -np.mean(std)
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

        sample["Reactor_Pressure"],
        sample["Feed_Temperature"],
        sample["Feed_Rate"],
        sample["Reactor_Temperature"],
        sample["Hydrogen_Flow"],
        sample["Catalyst_Loading"]

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

        "yield":[

            preds[0][1],
            preds[1][1],
          

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

        y="yield",

        title="yield Prediction Comparison"

    )

    st.plotly_chart(
        fig2,
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
