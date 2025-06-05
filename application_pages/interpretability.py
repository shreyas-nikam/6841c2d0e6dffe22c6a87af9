
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_interpretability():
    st.header("Interpretability Impact")
    st.write("Visualize the impact of model interpretability on risk exposure as a function of algorithmic assistance level (θ).")

    # --- Input parameters ---
    st.subheader("Input Parameters")

    theta_values = np.linspace(0, 1, 100)
    ch = st.number_input("Cost of Human Decision (c_h)", min_value=1000, max_value=100000, value=50000)
    cml = st.number_input("Cost of ML Decision (c_ml)", min_value=1000, max_value=100000, value=10000)

    # --- Cost Calculation based on Interpretability ---
    def calculate_cost(theta, ch, cml, curve_type):
        if curve_type == "Linear":
            return theta * cml + (1 - theta) * ch
        elif curve_type == "Concave":
            return (theta**0.5) * cml + (1 - (theta**0.5)) * ch
        elif curve_type == "Convex":
            return (theta**2) * cml + (1 - (theta**2)) * ch
        else:
            return np.zeros_like(theta)

    # --- Select curve type ---
    curve_type = st.selectbox("Curve Type", ["Linear", "Concave", "Convex"])

    # --- Calculate costs ---
    costs = calculate_cost(theta_values, ch, cml, curve_type)

    # --- Visualization ---
    st.subheader("Visualization")

    fig = px.line(x=theta_values, y=costs,
                  title=f"Contractual Risk Exposure vs. Algorithmic Assistance Level (θ) - {curve_type} Curve",
                  labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'Contractual Risk Exposure (c)'})
    st.plotly_chart(fig, use_container_width=True)
