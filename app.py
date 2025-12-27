import streamlit as st
import pandas as pd
import joblib

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="UIC Academic Decline Risk Predictor",
    page_icon="📘",
    layout="centered"
)

# =========================
# Load trained model
# =========================
data = joblib.load("model_tree.pkl")
model = data["model"]
accuracy = data["accuracy"]

features = [
    "Stress_level",
    "Burnout_Score",
    "Average_hours_sleep",
    "Class attendance",
    "Studying_Consistency_Encoded",
    "feeling_lost",
    "Schedule_effect",
    "Troubled_Modules",
    "Gender_Encoded"
]

# =========================
# Title & intro
# =========================
st.title("UIC Academic Performance Decline Risk Estimator")
st.write(
    """
    This interface estimates the **probability of academic performance decline**
    based on psychosocial and academic indicators.
    
    It is designed as a **research demonstration tool** to illustrate how
    machine learning models can support early academic risk awareness.
    """
)

st.divider()

# =========================
# Input section
# =========================
st.subheader("Student Profile")

col1, col2 = st.columns(2)

with col1:
    stress = st.slider("Perceived Stress Level (1–10)", 1, 10, 5)
    burnout = st.slider("Burnout Level (1–5)", 1, 5, 3)
    sleep = st.slider("Average Hours of Sleep per Night", 3.0, 9.0, 6.0)
    attendance = st.slider("Class Attendance (%)", 0, 100, 75)

with col2:
    study = st.selectbox(
        "Study Consistency",
        ["Daily", "Few times during the week", "Day before exam"]
    )
    lost = st.radio("Frequently feel lost during lectures?", ["No", "Yes"])
    schedule = st.radio("Perceived academic schedule as heavy?", ["No", "Yes"])
    modules = st.slider("Number of troubled modules", 0, 6, 1)
    gender = st.radio("Gender", ["Male", "Female"])

study_map = {
    "Daily": 0,
    "Few times during the week": 1,
    "Day before exam": 2
}

# =========================
# Prediction
# =========================
st.divider()

if st.button("Estimate Risk", use_container_width=True):
    input_data = pd.DataFrame([[
        stress,
        burnout,
        sleep,
        attendance,
        study_map[study],
        1 if lost == "Yes" else 0,
        1 if schedule == "Yes" else 0,
        modules,
        1 if gender == "Female" else 0
    ]], columns=features)

    probability = model.predict_proba(input_data)[0][1] * 100

    # Risk interpretation
    if probability < 30:
        risk_label = "Low"
    elif probability < 60:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    st.subheader("📊 Prediction Result")
    st.metric(
        "Estimated Probability of Academic Decline",
        f"{probability:.1f}%"
    )
    st.write(f"**Risk Level:** {risk_label}")

    # =========================
    # Explanation
    # =========================
    st.subheader("🔍 How the model reached this estimate")

    tree = model.named_steps["tree"]
    importances = tree.feature_importances_

    importance_df = pd.DataFrame({
        "Factor": features,
        "Relative Importance": importances
    }).sort_values(by="Relative Importance", ascending=False)

    st.write(
        "The decision tree model mainly relies on the following factors "
        "when estimating academic decline risk:"
    )
    st.dataframe(importance_df.head(5), use_container_width=True)

    # =========================
    # Model info & disclaimer
    # =========================
    st.divider()
    st.write(f"**Model accuracy (test data): {accuracy*100:.1f}%**")

    st.info(
        """
        **Important notice**

        This result is a **statistical prediction**, not a diagnosis.
        The model is trained exclusively on survey data collected
        from engineering students at Université Internationale de Casablanca (UIC).

        Predictions may not generalize to other institutions, populations,
        or individual situations. The tool is intended for **academic
        awareness and research demonstration purposes only**.
        """
    )
