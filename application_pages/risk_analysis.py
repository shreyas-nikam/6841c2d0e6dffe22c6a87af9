
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

def run_risk_analysis():
    st.header("Risk Analysis")
    st.write("Analyze the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) for different scenarios.")

    # --- Input parameters ---
    st.subheader("Input Parameters")

    n_patients = st.number_input("Number of Patients", min_value=100, max_value=10000, value=1000)
    mu = st.number_input("Mean Cost of False Positive (μ)", min_value=1000, max_value=100000, value=10000)
    sigma_mu = st.number_input("Standard Deviation of False Positive Cost (σ_μ)", min_value=100, max_value=10000, value=1000)
    M = st.number_input("Mean Cost of False Negative (M)", min_value=1000, max_value=100000, value=50000)
    sigma_M = st.number_input("Standard Deviation of False Negative Cost (σ_M)", min_value=100, max_value=10000, value=5000)
    confidence_level = st.slider("Confidence Level", min_value=0.01, max_value=0.99, value=0.95, step=0.01)
    theta_range = st.slider("Theta Range", min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.01)

    # --- Data Simulation (Simplified) ---
    @st.cache_data
    def simulate_errors(n_patients, mu, sigma_mu, M, sigma_M):
        fp_costs = np.random.normal(mu, sigma_mu, n_patients)
        fn_costs = np.random.normal(M, sigma_M, n_patients)
        return fp_costs, fn_costs

    fp_costs, fn_costs = simulate_errors(n_patients, mu, sigma_mu, M, sigma_M)
    total_costs = fp_costs + fn_costs

    # --- VaR and CVaR Calculation ---
    def calculate_var_cvar(costs, confidence_level):
        var = np.percentile(costs, confidence_level * 100)
        cvar = costs[costs >= var].mean()
        return var, cvar

    var, cvar = calculate_var_cvar(total_costs, confidence_level)

    # --- Impact of Theta ---
    theta_values = np.linspace(theta_range[0], theta_range[1], 100)
    expected_costs = []
    var_values = []
    cvar_values = []

    for theta in theta_values:
        # Simulate cost based on theta (algorithmic assistance level)
        simulated_costs = theta * fp_costs + (1 - theta) * fn_costs
        expected_costs.append(np.mean(simulated_costs))
        var, cvar = calculate_var_cvar(simulated_costs, confidence_level)
        var_values.append(var)
        cvar_values.append(cvar)

    # --- Visualization ---
    st.subheader("Risk Analysis Results")

    # Line chart: Expected Cost vs. Theta
    fig_cost = px.line(x=theta_values, y=expected_costs,
                       title="Expected Cost vs. Algorithmic Assistance Level (θ)",
                       labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'Expected Cost'})
    st.plotly_chart(fig_cost, use_container_width=True)

    # Line chart: VaR and CVaR vs. Theta
    fig_risk = px.line(x=theta_values, y=[var_values, cvar_values],
                       title=f"VaR and CVaR at {confidence_level:.2f} Confidence Level vs. Algorithmic Assistance Level (θ)",
                       labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'Risk'},
                       color_discrete_sequence=["#636EFA", "#EF553B"])
    fig_risk.add_trace(px.line(x=theta_values, y=var_values, labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'VaR'}).data[0])
    fig_risk.add_trace(px.line(x=theta_values, y=cvar_values, labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'CVaR'}).data[0])

    fig_risk.update_layout(legend_title_text='Risk Metric')

    st.plotly_chart(fig_risk, use_container_width=True)
