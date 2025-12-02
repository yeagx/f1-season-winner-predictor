import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="F1 Season Winner Predictor", layout="wide")

st.title("🏎️ F1 Season Winner Predictor")
st.write("Upload your F1 dataset and predict the next season's champion using machine learning.")


# ============================
# Upload File
# ============================

uploaded = st.file_uploader("Upload F1_flat_final.xlsx", type=["xlsx"])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success("File uploaded successfully!")

    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # ---------------------------
    # Build valid pit mask
    # ---------------------------
    valid_pit_mask = (df["pit_time"] >= 15) & (df["pit_time"] <= 40)

    # ---------------------------
    # Build driver-race table
    # ---------------------------
    driver_race = df.groupby(
        ["season", "race_name", "driver", "constructor"],
        as_index=False
    ).agg(
        total_pit_time=("pit_time", lambda x: x[valid_pit_mask.loc[x.index]].sum()),
        n_pitstops=("pit_time", lambda x: valid_pit_mask.loc[x.index].sum()),
        n_stints=("stint", "max"),
        points=("points", "max"),
        position=("position", "max"),
        laps_completed=("laps", "max"),
        avg_pit_time=("pit_time", lambda x: x[valid_pit_mask.loc[x.index]].mean()),
        avg_aggr=("driver_aggression_score", "max"),
        avg_pos_gain=("position_changes", "max")
    )

    # ---------------------------
    # Build Driver-Season Table
    # ---------------------------
    driver_season = driver_race.groupby(
        ["season", "driver", "constructor"], 
        as_index=False
    ).agg(
        races=("race_name", "nunique"),
        avg_finish=("position", "mean"),
        avg_points=("points", "mean"),
        total_points=("points", "sum"),
        avg_pitstops=("n_pitstops", "mean"),
        avg_pit_time=("avg_pit_time", "mean"),
        avg_aggr=("avg_aggr", "mean"),
        avg_pos_gain=("avg_pos_gain", "mean")
    )

    # Champion label
    driver_season["champion"] = driver_season.groupby("season")["total_points"] \
                                             .transform(lambda x: (x == x.max()).astype(int))


    st.write("### Driver-Season Data")
    st.dataframe(driver_season.head())

    # ============================
    # Encode Constructor
    # ============================
    le = LabelEncoder()
    driver_season["constructor_encoded"] = le.fit_transform(driver_season["constructor"])

    features = [
        "races", "avg_finish", "avg_points", "total_points",
        "avg_pitstops", "avg_pit_time",
        "avg_aggr", "avg_pos_gain",
        "constructor_encoded"
    ]

    X = driver_season[features]
    y = driver_season["champion"]

    # ============================
    # Train/Test Split
    # ============================
    test_size = st.slider("Test Size (%)", 10, 50, 25, 5) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ============================
    # Train Model
    # ============================
    st.subheader("Train Season Winner Model")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    st.write(f"### 📈 Model Accuracy: **{acc:.2f}**")

    # ============================
    # Feature Importance
    # ============================
    st.write("### Feature Importance")

    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(features, model.feature_importances_)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # ============================
    # Predict Next Season
    # ============================
    st.subheader("🔮 Predict Next Season Winner")

    last_season = driver_season["season"].max()
    next_season_df = driver_season[driver_season["season"] == last_season].copy()

    next_season_df["win_probability"] = model.predict_proba(next_season_df[features])[:,1]

    st.write("### Champion Prediction (Next Season)")
    st.dataframe(
        next_season_df[["driver", "constructor", "win_probability"]]
        .sort_values("win_probability", ascending=False)
    )
