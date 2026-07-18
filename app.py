
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

    return pd.read_excel(
        "dummy_plant_data.xlsx"
    )

df = load_data()

# =====================================================
# ANOMALY MODEL
# =====================================================

@st.cache_resource
def train_anomaly_model(df):

    features = [

        "Reactor_Pressure",
        "Feed_Temperature",
        "Feed_Rate",
        "Reactor_Temperature",
        "Hydrogen_Flow",
        "Catalyst_Loading"

    ]

    model = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    model.fit(df[features])

    return model

iso_model = train_anomaly_model(df)

feature_names = [

    "Reactor_Pressure",
    "Feed_Temperature",
    "Feed_Rate",
    "Reactor_Temperature",
    "Hydrogen_Flow",
    "Catalyst_Loading"

]

# =====================================================
# LIVE DATA FOR DEMO
# =====================================================

live_data = df.sample(1)

score = iso_model.decision_function(
    live_data[feature_names]
)[0]

prediction = iso_model.predict(
    live_data[feature_names]
)[0]

health = max(
    0,
    min(
        100,
        (score + 0.5) * 100
    )
)

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
    target_productivity

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

            (pred[1]-target_productivity)**2

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
tab1,tab2,tab3,tab4= st.tabs(

    [

        "Live PFD",

        "Historical Data",

        "Optimizer",

        "Model diagnostics"


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
        "productivity_pred": float(current["productivity"])
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


    st.markdown(
    f"""
    <div style="
    width:100%;
    height:450px;
    position:relative;
    border:1px solid #444;
    border-radius:10px;
    ">

    <!-- FEED LINE -->

    <div style="
    position:absolute;
    left:80px;
    top:190px;
    width:220px;
    border-top:4px solid #4FC3F7;
    ">
    </div>

    <!-- FEED VALUES -->

    <div style="
    position:absolute;
    left:100px;
    top:200px;
    color:blue;
    ">
    <b>FEED</b><br>
    T = {st.session_state.feed_temp:.1f} °C<br>
    Flow rate = {st.session_state.feed_rate:.1f} m3/h<br>
    </div>

    <!-- HYDROGEN LINE -->

    <div style="
    position:absolute;
    left:360px;
    top:40px;
    width:4px;
    height:60px;
    background:#4FC3F7;
    ">
    </div>

    <!-- HYDROGEN VALUES -->

    <div style="
    position:absolute;
    left:210px;
    top:20px;
    color:blue;
    text-align:center;
    ">
    <b>HYDROGEN</b><br>
    Flow rate = {st.session_state.hydrogen_flow:.1f} m3/h<br>
    </div>

    <!-- REACTOR -->

    <div style="
    position:absolute;
    left:300px;
    top:100px;
    width:120px;
    height:180px;
    border:4px solid #4FC3F7;
    border-radius:40px;
    text-align:center;
    color:blue;
    padding-top:20px;
    font-size:14px;
    ">

    <b>REACTOR</b>

    <br><br>

    T = {st.session_state.reactor_temp:.1f} °C<br>
    P = {st.session_state.reactor_pressure:.1f} bar

    </div>

    <!-- CATALYST LINE -->

    <div style="
    position:absolute;
    left:360px;
    top:280px;
    width:4px;
    height:60px;
    background:#4FC3F7;
    ">
    </div>

    <!-- CATALYST VALUES -->

    <div style="
    position:absolute;
    left:250px;
    top:330px;
    color:blue;
    text-align:center;
    ">
    <b>CATALYST</b><br>
    Load = {st.session_state.catalyst_loading:.2f} Kg/h<br>
    </div>

    <!-- PRODUCT LINE -->

    <div style="
    position:absolute;
    left:420px;
    top:190px;
    width:220px;
    border-top:4px solid #4FC3F7;
    ">
    </div>

    <!-- PRODUCT VALUES -->

    <div style="
    position:absolute;
    left:500px;
    top:200px;
    color:blue;
    ">
    <b>PRODUCT</b><br>
    MFI = {st.session_state.mfi_pred:.2f}<br>
    productivity = {st.session_state.productivity_pred:.2f}
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
        st.session_state.productivity_pred = float(pred[1])

        confidence = np.exp(
            -np.mean(std)
        ) * 100

        st.success(
            f"Prediction Completed | Confidence = {confidence:.1f}%"
        )

        mfi = st.session_state.mfi_pred
        productivity_value = st.session_state.productivity_pred

        




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

        target_productivity = st.number_input(

            "Target productivity",

            value=90.0,

            key="opt_target_productivity"

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

                target_productivity

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

                "Predicted productivity",

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

        "productivity":[

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

        y="productivity",

        title="productivity Prediction Comparison"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

st.caption(
    "Advanced Process Control Dashboard | Ensemble ML Optimizer"
)
