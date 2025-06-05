
## Algorithmic Insurance Premium Calculator: Technical Specifications

### Overview

This document outlines the technical specifications for a Streamlit application designed to calculate insurance premiums for algorithmic insurance contracts. The application focuses on medical malpractice lawsuits related to malignant tumor detection, drawing inspiration from the attached research paper. The primary goal is to allow users to interact with key parameters and visualize their impact on the calculated premium, promoting a deeper understanding of algorithmic insurance concepts.

### Step-by-Step Generation Process

1.  **Set up Environment:**
    *   Create a new Python environment (recommended using `venv` or `conda`) to manage dependencies.
    *   Install the required libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, and potentially `matplotlib` or `plotly` for visualizations.

2.  **Import Libraries:**
    *   In your Python script (e.g., `app.py`), import the necessary libraries:

```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
#Optional, if want to use plotly
#import plotly.express as px
```

3.  **Create Input Forms:**
    *   Use Streamlit widgets to create input forms for the following parameters:

```python
st.header("Algorithmic Insurance Premium Calculator")

st.subheader("Input Parameters")
litigation_cost_false_negative = st.number_input("Litigation Cost (False Negative) - L", min_value=0, value=500000)
litigation_cost_false_positive = st.number_input("Litigation Cost (False Positive) - K", min_value=0, value=100000)
num_patients = st.number_input("Number of Patients (N)", min_value=1, value=100)
contract_price_upper_bound = st.number_input("Contract Price Upper Bound (Hp)", min_value=1000, value=50000)
classification_threshold = st.slider("Classification Threshold (τ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
```

4.  **Load/Generate Dataset**
    * Since the requirement is to generate a synthetic dataset, create a function that does this.

```python
def generate_synthetic_data(num_samples=1000):
    # Example features (replace with more relevant features later if needed)
    data = {
        'radius_mean': np.random.rand(num_samples) * 20,  # Example range
        'texture_mean': np.random.rand(num_samples) * 30,  # Example range
        'perimeter_mean': np.random.rand(num_samples) * 150, # Example range
        'area_mean': np.random.rand(num_samples) * 1200, # Example range
        'smoothness_mean': np.random.rand(num_samples) * 0.2, # Example range
        'compactness_mean': np.random.rand(num_samples) * 0.4,  # Example range
        'concavity_mean': np.random.rand(num_samples) * 1, # Example range
        'concave points_mean': np.random.rand(num_samples) * 0.2, # Example range
        'symmetry_mean': np.random.rand(num_samples) * 0.4,  # Example range
        'fractal_dimension_mean': np.random.rand(num_samples) * 0.1, # Example range
        'malignant': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]) # Example: 40% malignant
    }
    df = pd.DataFrame(data)
    return df

df = generate_synthetic_data()
st.subheader("Synthetic Data Sample")
st.dataframe(df.head())
```

5.  **Train ML Model (Optional)**
    *   If the requirement needs an ML model, then the synthetic data will be used to train a simple model.

```python
# Split data into training and testing sets
X = df.drop('malignant', axis=1)
y = df['malignant']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

st.subheader("Model Training Complete")
```

6.  **Implement Premium Calculation:**
    *   Translate Equation 3 (or 4 if robust optimization is desired) from Section 4 of the document into Python code.

```python
def calculate_premium(L, K, N, tau, model, X_test, y_test):
    # Predict probabilities on the test set
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the classification threshold
    predictions = (probabilities >= tau).astype(int)

    # Calculate the confusion matrix
    tp = np.sum((predictions == 1) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))

    # Calculate expected loss based on the confusion matrix
    expected_loss = (fn / len(y_test)) * L + (fp / len(y_test)) * K
    total_expected_loss = N * expected_loss

    return total_expected_loss

premium = calculate_premium(litigation_cost_false_negative, litigation_cost_false_positive, num_patients, classification_threshold, model, X_test, y_test)

st.subheader("Calculated Premium")
st.write(f"Estimated Premium: ${premium:,.2f}")
```

7.  **Implement Sensitivity Analysis:**

```python
st.subheader("Sensitivity Analysis")

thresholds = np.arange(0.1, 1.0, 0.05)
premiums = [calculate_premium(litigation_cost_false_negative, litigation_cost_false_positive, num_patients, t, model, X_test, y_test) for t in thresholds]

fig, ax = plt.subplots()
ax.plot(thresholds, premiums)
ax.set_xlabel("Classification Threshold (τ)")
ax.set_ylabel("Estimated Premium")
ax.set_title("Premium vs. Classification Threshold")
st.pyplot(fig)

#An alternative to pyplot would be plotly:
#fig = px.line(x=thresholds, y=premiums, labels={'x':'Classification Threshold (τ)', 'y':'Estimated Premium'}, title='Premium vs. Classification Threshold')
#st.plotly_chart(fig)
```

8.  **Display Results and Visualizations:**
    *   Display the calculated premium using `st.write()`.
    *   Generate interactive charts to visualize the relationship between the classification threshold (τ) and the calculated premium. Use `matplotlib` or `plotly` to create these charts.

9.  **Add Documentation:**

```python
st.sidebar.header("Documentation")
st.sidebar.write("This application calculates the insurance premium for algorithmic insurance contracts...")
st.sidebar.write("Adjust the parameters on the left to see how they affect the premium.")
```

### Important Definitions, Examples, and Formulae

*   **Algorithmic Insurance:** Insurance contracts designed to protect against the financial risks associated with errors made by machine learning algorithms.

*   **Conditional Value-at-Risk (CVaR):** A risk measure that quantifies the expected loss given that the loss exceeds a certain threshold. In the context of this application, CVaR estimates the potential financial loss an insurance company might face due to medical malpractice lawsuits arising from errors in tumor detection algorithms.
    *   Formula:

    ```latex
    CVaR_{\beta}(X) = E[X | X \geq VaR_{\beta}(X)]
    ```

    Where:
    *   `X` is the random variable representing the loss.
    *   `\beta` is the confidence level.
    *   `VaR_{\beta}(X)` is the Value-at-Risk at confidence level `\beta`.

    *Example:* A CVaR at 95% confidence estimates the expected loss, given that the loss is in the worst 5% of cases.

*   **Value-at-Risk (VaR):** A risk measure that estimates the maximum loss expected over a specific time horizon at a given confidence level.
    *   Formula:

    ```latex
    P(X \leq VaR_{\beta}(X)) = \beta
    ```

    Where:
    *   `X` is the random variable representing the loss.
    *   `\beta` is the confidence level.

    *Example:* A VaR at 95% confidence level means that there is a 95% probability that the loss will not exceed the VaR value.

*   **Classification Threshold (τ):** A threshold used to classify data points as positive or negative based on the output of a machine learning model. In this application, the threshold determines whether a patient is classified as having a malignant tumor.

    *Example:* If a model outputs a probability of 0.7 that a patient has a tumor, and the classification threshold is 0.5, the patient is classified as having a tumor.

*   **Sensitivity:** The ability of a model to correctly identify positive cases (i.e., patients with tumors).
    *   Formula:

    ```latex
    Sensitivity = \frac{True Positives}{True Positives + False Negatives}
    ```

*   **Specificity:** The ability of a model to correctly identify negative cases (i.e., patients without tumors).
    *   Formula:

    ```latex
    Specificity = \frac{True Negatives}{True Negatives + False Positives}
    ```

*   **Equation 3 (from the research paper, representing the basic optimization formulation):**

    ```latex
    \min_{\alpha, x} \alpha + \nu \sum_{j=1}^{J} z_j \\
    s.t. \quad z_j \geq \sum_{p=1}^{P} \max\{0, (y_{pj} - x_p)\} - \alpha, \quad \forall j \in [J] \\
    \quad z_j \geq 0, \quad \forall j \in [J] \\
    \quad x_p \leq H_p, \quad \forall p \in [P]
    ```

    Where:

    *   `α` is the Value-at-Risk (VaR).
    *   `\nu = \frac{1}{1 - \beta}` where `\beta` is the confidence level.
    *   `z_j` represents the excess loss beyond VaR in scenario `j`.
    *   `y_{pj}` is the loss in premium category `p` under scenario `j`.
    *   `x_p` is the insurance premium for category `p`.
    *   `H_p` is the upper bound on the premium for category `p`.
    *   `[J]` is the set of all scenarios.
    *   `[P]` is the set of all premium categories.
* **Equation 7 (from the research paper, representing the Claim Cost):**
```latex
S = (1- \kappa_{\tau})K + (1 - \lambda_{\tau})L
```
Where:
* S is the claim cost
* K is the litigation cost of a false positive
* L is the litigation cost of a false negative
* κ is the specificity
* λ is the sensitivity

### Libraries and Tools

*   **Streamlit:** Used for building the interactive user interface, handling user inputs, and displaying results and visualizations.
*   **Pandas:** Used for data manipulation, creating dataframes for synthetic datasets, and potentially for loading data from external files.
*   **NumPy:** Used for numerical computations, generating synthetic data, and performing calculations for premium estimation and sensitivity analysis.
*   **Scikit-learn:** Used for training a machine learning model (e.g., Random Forest) to predict the probability of malignant tumors.
*   **Matplotlib/Plotly:** Used for creating interactive charts to visualize the relationship between different parameters and the calculated premium. `matplotlib` is a static plotting library, while `plotly` allows for dynamic interactive plots.

### Relation to Paper Concepts

This application directly relates to Section 4 of the research paper, which discusses the quantitative framework for estimating the risk exposure of algorithmic insurance contracts. The application implements the optimization formulation presented in Equation 3 (or 4) to calculate the insurance premium.

Furthermore, the sensitivity analysis of the classification threshold (τ), as visualized in Figure 1 of the paper, is replicated in the application, allowing users to understand the trade-offs between sensitivity and specificity and their impact on the overall risk and premium.
