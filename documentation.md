id: 6841c2d0e6dffe22c6a87af9_documentation
summary: Algorithmic Insurance - Medical malpractice case Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Algorithmic Insurance - Medical Malpractice Case: A Codelab

This codelab provides a comprehensive guide to understanding and exploring the "Algorithmic Insurance - Medical Malpractice Case" Streamlit application. This application simulates medical malpractice risk in breast cancer detection, with a focus on how algorithmic assistance can impact this risk.  You will learn how to use the application to explore different levels of human-AI collaboration and visualize the effects on risk exposure.  The application is based on concepts from "Algorithmic Insurance" by Bertsimas and Orfanoudaki (2022).

**Key Concepts Covered:**

*   Binary Classification
*   Classification Threshold
*   Sensitivity & Specificity
*   False Positives & False Negatives
*   Value at Risk (VaR) & Conditional Value at Risk (CVaR)
*   Algorithmic Assistance Level (θ)
*   Expected Cost

## Understanding the Core Application (app.py)
Duration: 00:05

The `app.py` file serves as the main entry point for the Streamlit application. It handles the overall layout, navigation, and routing to different sections of the application.

```python
import streamlit as st

st.set_page_config(page_title="Algorithmic Insurance - Medical Malpractice Case", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("Algorithmic Insurance - Medical Malpractice Case")
st.divider()

st.markdown("""
This application simulates medical malpractice risk in breast cancer detection, focusing on the role of algorithmic assistance. It allows users to explore different levels of human-AI collaboration and visualize the impact on risk exposure. The application is inspired by and grounded in the concepts presented in the document "Algorithmic Insurance" by Bertsimas and Orfanoudaki (2022), specifically Section 3 (case study of medical liability in breast cancer detection) and Section 6 (incorporating interpretability/human-machine interaction into the pricing strategy).

**Key Concepts:**

*   **Binary Classification:** A classification task with two possible outcomes (e.g., Benign or Malignant).
*   **Classification Threshold (τ):** A value used to classify data points based on the output of a binary classification model.
*   **Sensitivity (Recall):** The ability of a model to correctly identify positive cases.  Mathematically: $\text{Sensitivity} = \frac{TP}{TP + FN}$.
*   **Specificity:** The ability of a model to correctly identify negative cases.  Mathematically: $\text{Specificity} = \frac{TN}{TN + FP}$.
*   **False Positive (FP):** Incorrectly classifying a negative case as positive.
*   **False Negative (FN):** Incorrectly classifying a positive case as negative.
*   **Value at Risk (VaR):**  The maximum loss with a specified confidence level.
*   **Conditional Value at Risk (CVaR):** The expected loss if the VaR confidence level is exceeded.
*   **Algorithmic Assistance Level (θ):** A parameter representing the degree of human-AI collaboration in decision-making (0 to 1).
*   **Expected Cost (E(C)):** The estimated financial loss due to medical malpractice claims, calculated as: $E(C) = N((1 − κ_τ)μ + (1 − λ_τ)M)$

**Formulae:**

*   $\text{Sensitivity} = \frac{TP}{TP + FN}$
*   $\text{Specificity} = \frac{TN}{TN + FP}$
*   $E(C) = N((1 − κ_τ)μ + (1 − λ_τ)M)$
*   $c = \theta c_{ml} + (1-\theta)c_h$

The application provides interactive visualizations to explore the relationship between these factors and the overall risk exposure.
""")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Simulator", "Risk Analysis", "Interpretability"])

if page == "Simulator":
    from application_pages.simulator import run_simulator
    run_simulator()
elif page == "Risk Analysis":
    from application_pages.risk_analysis import run_risk_analysis
    run_risk_analysis()
elif page == "Interpretability":
    from application_pages.interpretability import run_interpretability
    run_interpretability()
# Your code ends

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity.")
```

**Explanation:**

1.  **Import Streamlit:**  The code starts by importing the Streamlit library (`import streamlit as st`).
2.  **Page Configuration:** `st.set_page_config` sets the title and layout of the Streamlit app.  `layout="wide"` makes the app use the full width of the browser window.
3.  **Sidebar Elements:** The sidebar is populated with the QuantUniversity logo and a divider.
4.  **Title and Introduction:** The main area of the application displays the title and a description of the application's purpose, key concepts, and relevant formulae. The `st.markdown` function allows you to include formatted text, including HTML and LaTeX-style mathematical notation.
5.  **Navigation:** A `st.sidebar.selectbox` creates a dropdown menu in the sidebar, allowing users to choose between three pages: "Simulator", "Risk Analysis", and "Interpretability".
6.  **Page Routing:** The `if/elif/else` block handles the navigation.  When a user selects a page from the dropdown, the corresponding function is imported from the `application_pages` directory and executed (e.g., `run_simulator()`).
7.  **Footer:** A footer with copyright information and a disclaimer is added at the bottom of the page.

## Exploring the Medical Malpractice Risk Simulator (simulator.py)
Duration: 00:15

This section walks through the functionality of the "Simulator" page, which allows you to simulate medical malpractice risk based on various input parameters.

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_simulator():
    st.header("Medical Malpractice Risk Simulator")
    st.write("Simulate the impact of algorithmic assistance on medical malpractice risk.")

    #  Input parameters 
    st.subheader("Input Parameters")

    n_patients = st.number_input("Number of Patients", min_value=100, max_value=10000, value=1000)
    tau = st.slider("Classification Threshold (τ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    theta = st.slider("Algorithmic Assistance Level (θ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    mu = st.number_input("Mean Cost of False Positive (μ)", min_value=1000, max_value=100000, value=10000)
    sigma_mu = st.number_input("Standard Deviation of False Positive Cost (σ_μ)", min_value=100, max_value=10000, value=1000)
    M = st.number_input("Mean Cost of False Negative (M)", min_value=1000, max_value=100000, value=50000)
    sigma_M = st.number_input("Standard Deviation of False Negative Cost (σ_M)", min_value=100, max_value=10000, value=5000)

    #  Data Simulation 
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

    #  Model Simulation 
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

    #  Cost Simulation 
    expected_cost = n_patients * ((1 - specificity) * mu + (1 - sensitivity) * M)

    #  Results 
    st.subheader("Results")
    st.write(f"Sensitivity: {sensitivity:.2f}")
    st.write(f"Specificity: {specificity:.2f}")
    st.write(f"Expected Cost: ${expected_cost:,.2f}")

    #  Visualization 
    st.subheader("Visualization")
    error_data = pd.DataFrame({
        'Error Type': ['False Positive', 'False Negative'],
        'Cost': [FP * mu, FN * M]
    })

    fig = px.bar(error_data, x='Error Type', y='Cost', color='Error Type',
                 title='Cost by Error Type')
    st.plotly_chart(fig, use_container_width=True)
```

**Explanation:**

1.  **Imports:** The code imports necessary libraries: `streamlit` for the user interface, `pandas` for data manipulation, `numpy` for numerical operations, and `plotly.express` for creating visualizations.
2.  **`run_simulator()` Function:** This function encapsulates the logic for the simulator page.
3.  **Input Parameters:** The code uses Streamlit widgets (`st.number_input`, `st.slider`) to allow users to define the parameters for the simulation:
    *   `n_patients`: Number of patients in the simulation.
    *   `tau`: Classification threshold for the model.
    *   `theta`: Algorithmic assistance level.
    *   `mu`: Mean cost of a false positive.
    *   `sigma_mu`: Standard deviation of the false positive cost.
    *   `M`: Mean cost of a false negative.
    *   `sigma_M`: Standard deviation of the false negative cost.
4.  **Data Simulation:** The `simulate_data` function generates a sample dataset with patient data (radius, texture, outcome).
    *   `@st.cache_data` decorator: This decorator tells Streamlit to cache the results of the `simulate_data` function. This means that Streamlit will only run the function once, and then store the results in a cache. If the function is called again with the same inputs, Streamlit will simply return the cached results instead of running the function again. This can significantly improve the performance of Streamlit apps that perform expensive computations.
5.  **Model Simulation:** The `simulate_model_performance` function simulates the performance of a binary classification model. It calculates the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), as well as the sensitivity and specificity of the model based on a random prediction.  **Note:** This is a placeholder for a real model; in a production application, you would replace this with a trained machine learning model.
6.  **Cost Simulation:** The code calculates the expected cost based on the simulated model performance and the input cost parameters.
7.  **Results:** The calculated sensitivity, specificity, and expected cost are displayed to the user using `st.write`.
8.  **Visualization:** A bar chart is created using `plotly.express` to visualize the cost associated with false positives and false negatives.  This chart helps users understand the relative impact of each type of error.

## Analyzing Risk with VaR and CVaR (risk_analysis.py)
Duration: 00:15

The "Risk Analysis" page focuses on calculating and visualizing Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) to assess the potential financial risks.

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

def run_risk_analysis():
    st.header("Risk Analysis")
    st.write("Analyze the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) for different scenarios.")

    #  Input parameters 
    st.subheader("Input Parameters")

    n_patients = st.number_input("Number of Patients", min_value=100, max_value=10000, value=1000)
    mu = st.number_input("Mean Cost of False Positive (μ)", min_value=1000, max_value=100000, value=10000)
    sigma_mu = st.number_input("Standard Deviation of False Positive Cost (σ_μ)", min_value=100, max_value=10000, value=1000)
    M = st.number_input("Mean Cost of False Negative (M)", min_value=1000, max_value=100000, value=50000)
    sigma_M = st.number_input("Standard Deviation of False Negative Cost (σ_M)", min_value=100, max_value=10000, value=5000)
    confidence_level = st.slider("Confidence Level", min_value=0.01, max_value=0.99, value=0.95, step=0.01)
    theta_range = st.slider("Theta Range", min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.01)

    #  Data Simulation (Simplified) 
    @st.cache_data
    def simulate_errors(n_patients, mu, sigma_mu, M, sigma_M):
        fp_costs = np.random.normal(mu, sigma_mu, n_patients)
        fn_costs = np.random.normal(M, sigma_M, n_patients)
        return fp_costs, fn_costs

    fp_costs, fn_costs = simulate_errors(n_patients, mu, sigma_mu, M, sigma_M)
    total_costs = fp_costs + fn_costs

    #  VaR and CVaR Calculation 
    def calculate_var_cvar(costs, confidence_level):
        var = np.percentile(costs, confidence_level * 100)
        cvar = costs[costs >= var].mean()
        return var, cvar

    var, cvar = calculate_var_cvar(total_costs, confidence_level)

    #  Impact of Theta 
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

    #  Visualization 
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
```

**Explanation:**

1.  **Imports:** Similar to the simulator, this code imports `streamlit`, `pandas`, `numpy`, and `plotly.express`.  It also imports `scipy.stats.norm` for statistical calculations (although it's not directly used in this specific code, it's often used in VaR/CVaR calculations).
2.  **`run_risk_analysis()` Function:**  Encapsulates the risk analysis logic.
3.  **Input Parameters:** The code takes several input parameters, including the number of patients, mean and standard deviation of false positive and false negative costs, the confidence level for VaR/CVaR, and a range for the algorithmic assistance level (theta).
4.  **Data Simulation:** The `simulate_errors` function generates simulated costs for false positives and false negatives using a normal distribution.
5.  **VaR and CVaR Calculation:** The `calculate_var_cvar` function calculates the VaR (Value at Risk) and CVaR (Conditional Value at Risk) for a given set of costs and a specified confidence level.  VaR represents the maximum expected loss at the given confidence level, while CVaR represents the expected loss *given that* the VaR is exceeded.
6.  **Impact of Theta:** The code iterates through a range of theta values (algorithmic assistance levels).  For each theta, it calculates the simulated costs (weighted combination of false positive and false negative costs), the expected cost, and the VaR and CVaR.
7.  **Visualization:** The code generates two line charts:
    *   **Expected Cost vs. Theta:** Shows how the expected cost changes as the algorithmic assistance level varies.
    *   **VaR and CVaR vs. Theta:** Shows how the VaR and CVaR change as the algorithmic assistance level varies.  This is crucial for understanding how algorithmic assistance impacts the tail risk (the risk of extreme losses).

## Exploring the Impact of Interpretability (interpretability.py)
Duration: 00:10

The "Interpretability" page focuses on visualizing the impact of model interpretability on risk exposure as a function of the algorithmic assistance level (θ).

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_interpretability():
    st.header("Interpretability Impact")
    st.write("Visualize the impact of model interpretability on risk exposure as a function of algorithmic assistance level (θ).")

    #  Input parameters 
    st.subheader("Input Parameters")

    theta_values = np.linspace(0, 1, 100)
    ch = st.number_input("Cost of Human Decision (c_h)", min_value=1000, max_value=100000, value=50000)
    cml = st.number_input("Cost of ML Decision (c_ml)", min_value=1000, max_value=100000, value=10000)

    #  Cost Calculation based on Interpretability 
    def calculate_cost(theta, ch, cml, curve_type):
        if curve_type == "Linear":
            return theta * cml + (1 - theta) * ch
        elif curve_type == "Concave":
            return (theta**0.5) * cml + (1 - (theta**0.5)) * ch
        elif curve_type == "Convex":
            return (theta**2) * cml + (1 - (theta**2)) * ch
        else:
            return np.zeros_like(theta)

    #  Select curve type 
    curve_type = st.selectbox("Curve Type", ["Linear", "Concave", "Convex"])

    #  Calculate costs 
    costs = calculate_cost(theta_values, ch, cml, curve_type)

    #  Visualization 
    st.subheader("Visualization")

    fig = px.line(x=theta_values, y=costs,
                  title=f"Contractual Risk Exposure vs. Algorithmic Assistance Level (θ) - {curve_type} Curve",
                  labels={'x': 'Algorithmic Assistance Level (θ)', 'y': 'Contractual Risk Exposure (c)'})
    st.plotly_chart(fig, use_container_width=True)
```

**Explanation:**

1.  **Imports:**  Standard imports for Streamlit, data manipulation, and visualization.
2.  **`run_interpretability()` Function:** Contains the logic for the interpretability analysis.
3.  **Input Parameters:** The code takes the cost of a human decision (`ch`) and the cost of a machine learning decision (`cml`) as input.
4.  **Cost Calculation:** The `calculate_cost` function calculates the contractual risk exposure based on the algorithmic assistance level (`theta`), the costs of human and machine learning decisions, and the selected curve type. The `curve_type` parameter allows users to explore different relationships between algorithmic assistance and risk exposure. The code implements three curve types:
    *   **Linear:** A simple linear interpolation between the cost of a human decision and the cost of a machine learning decision.
    *   **Concave:**  This curve type assumes that the benefit of algorithmic assistance diminishes as the assistance level increases.
    *   **Convex:** This curve type assumes that the benefit of algorithmic assistance increases as the assistance level increases.
5.  **Curve Type Selection:** A `st.selectbox` allows the user to choose between "Linear", "Concave", and "Convex" curve types.
6.  **Visualization:** A line chart displays the contractual risk exposure as a function of the algorithmic assistance level, based on the selected curve type.

## Application Architecture

The application follows a modular structure, separating the main application logic from the individual page implementations.

```
app.py (Main Application)
├── application_pages/
│   ├── simulator.py (Medical Malpractice Risk Simulator)
│   ├── risk_analysis.py (VaR and CVaR Analysis)
│   └── interpretability.py (Interpretability Impact Visualization)
```

**Data Flow:**

1.  The user interacts with the Streamlit application through the browser.
2.  The `app.py` file handles routing based on user selection in the sidebar.
3.  The selected page's corresponding Python file (`simulator.py`, `risk_analysis.py`, or `interpretability.py`) is executed.
4.  Each page takes input parameters from the user through Streamlit widgets.
5.  The page performs calculations and simulations based on the input parameters.
6.  The page generates visualizations using Plotly Express.
7.  The results and visualizations are displayed to the user in the Streamlit application.

<aside class="positive">
This codelab provided a detailed walkthrough of the Algorithmic Insurance Medical Malpractice Case application.  Experiment with different input parameters to gain a deeper understanding of the relationships between algorithmic assistance, risk exposure, and interpretability.
</aside>
