
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
