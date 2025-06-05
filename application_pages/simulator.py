
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_simulator():
    st.header("Medical Malpractice Risk Simulator")
    st.write("Simulate the impact of algorithmic assistance on medical malpractice risk.")

    # --- Input parameters ---
    st.subheader("Input Parameters")

    n_patients = st.number_input("Number of Patients", min_value=100, max_value=10000, value=1000)
    tau = st.slider("Classification Threshold (τ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    theta = st.slider("Algorithmic Assistance Level (θ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    mu = st.number_input("Mean Cost of False Positive (μ)", min_value=1000, max_value=100000, value=10000)
    sigma_mu = st.number_input("Standard Deviation of False Positive Cost (σ_μ)", min_value=100, max_value=10000, value=1000)
    M = st.number_input("Mean Cost of False Negative (M)", min_value=1000, max_value=100000, value=50000)
    sigma_M = st.number_input("Standard Deviation of False Negative Cost (σ_M)", min_value=100, max_value=10000, value=5000)

    # --- Data Simulation ---
    @st.cache_data
    def simulate_data(n_patients):
        # This is a simplified simulation. Replace with more realistic data generation if needed.
        data = pd.DataFrame({
            'radius': np.random.normal(15, 3, n_patients),
            'texture': np.random.normal(20, 5, n_patients),
            'outcome': np.random.choice([0, 1], n_patients, p=[0.7, 0.3])  # 0: Benign, 1: Malignant
        })
        return data

    data = simulate_data(n_patients)

    # --- Model Simulation ---
    # This is a placeholder. In a real application, you would use a trained model.
    def simulate_model_performance(data, tau):
        # Simulate model predictions (replace with actual model predictions)
        data['prediction'] = np.random.rand(len(data))
        data['predicted_outcome'] = (data['prediction'] > tau).astype(int)

        TP = len(data[(data['outcome'] == 1) & (data['predicted_outcome'] == 1)])
        TN = len(data[(data['outcome'] == 0) & (data['predicted_outcome'] == 0)])
        FP = len(data[(data['outcome'] == 0) & (data['predicted_outcome'] == 1)])
        FN = len(data[(data['outcome'] == 1) & (data['predicted_outcome'] == 0)])

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

        return sensitivity, specificity, TP, TN, FP, FN

    sensitivity, specificity, TP, TN, FP, FN = simulate_model_performance(data.copy(), tau)

    # --- Cost Simulation ---
    expected_cost = n_patients * ((1 - specificity) * mu + (1 - sensitivity) * M)

    # --- Results ---
    st.subheader("Results")
    st.write(f"Sensitivity: {sensitivity:.2f}")
    st.write(f"Specificity: {specificity:.2f}")
    st.write(f"Expected Cost: ${expected_cost:,.2f}")

    # --- Visualization ---
    st.subheader("Visualization")
    error_data = pd.DataFrame({
        'Error Type': ['False Positive', 'False Negative'],
        'Cost': [FP * mu, FN * M]
    })

    fig = px.bar(error_data, x='Error Type', y='Cost', color='Error Type',
                 title='Cost by Error Type')
    st.plotly_chart(fig, use_container_width=True)
