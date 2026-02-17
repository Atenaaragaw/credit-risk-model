import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

# 1. Setup Page Config
st.set_page_config(page_title="PrecisionCredit Decision Support", layout="wide")

# 2. Load the Pipeline
@st.cache_resource
def load_model():
    return joblib.load("./model/credit_risk_model.joblib")

pipeline = load_model()
model = pipeline[-1]
preprocessor = pipeline[:-1]

st.title("🛡️ PrecisionCredit: Decision Support System")
st.markdown("---")

# 3. Sidebar for Input
st.sidebar.header("Customer Application Details")

def user_input_features():
    # Keep the UI friendly, but map to the expected internal names
    recency = st.sidebar.slider("Recency (Days)", 0, 365, 30)
    frequency = st.sidebar.slider("Frequency (Transactions)", 1, 100, 10)
    
    # Map 'Monetary' to 'total_amount' if that's what the model expects
    total_amount = st.sidebar.number_input("Total Amount", 0.0, 1000000.0, 50000.0)
    avg_amount = total_amount / frequency if frequency > 0 else 0
    std_amount = avg_amount * 0.1 # Approximation for the UI
    
    transaction_count = st.sidebar.number_input("Transaction Count", 1, 500, 20)
    currency = st.sidebar.selectbox("Currency", ["TZS", "UGX", "KES"])
    country = st.sidebar.selectbox("Country", ["Tanzania", "Uganda", "Kenya"])
    provider = st.sidebar.selectbox("Provider", ["ProviderA", "ProviderB"])
    product = st.sidebar.selectbox("Product", ["ProductX", "ProductY"])
    category = st.sidebar.selectbox("Category", ["financial_services", "airtime"])
    channel = st.sidebar.selectbox("Channel", ["Web", "Mobile"])
    strategy = st.sidebar.selectbox("Strategy", [0, 1, 2, 4])

    # Ensure these keys match the ERROR message perfectly
    data = {
        "Recency": recency,
        "Frequency": frequency,
        "total_amount": total_amount,     # Fixed name
        "avg_amount": avg_amount,         # Fixed name
        "std_amount": std_amount,         # Fixed name
        "transaction_count": transaction_count,
        "CurrencyCode": currency,
        "CountryCode": country,
        "ProviderId": provider,
        "ProductId": product,
        "ProductCategory": category,
        "ChannelId": channel,
        "PricingStrategy": strategy
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. Main Section: Prediction
col1, col2 = st.columns(2)

with col1:
    st.subheader("Application Summary")
    st.write(input_df)
    
    # Prediction
    prob = pipeline.predict_proba(input_df)[0][1]
    risk_level = "HIGH" if prob > 0.5 else "LOW"
    
    st.metric(label="Risk Probability", value=f"{prob:.2%}")
    if risk_level == "HIGH":
        st.error(f"Decision: {risk_level} RISK")
    else:
        st.success(f"Decision: {risk_level} RISK")

# 5. Explainability Section
with col2:
    st.subheader("Audit Transparency (SHAP)")
    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    explainer = shap.LinearExplainer(model, X_transformed) # Since we use Logistic Regression
    shap_values = explainer.shap_values(X_transformed)

    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)