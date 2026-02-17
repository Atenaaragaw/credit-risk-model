import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Load your pipeline
pipeline = joblib.load("./model/credit_risk_model.joblib")
model = pipeline[-1] 
preprocessor = pipeline[:-1] 

# 2. Load sample data
X_raw = pd.read_csv("./data/sample_data.csv")
X_transformed = preprocessor.transform(X_raw)

# 3. Handle Sparse Matrix (Fixes the TypeError)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# 4. Initialize LinearExplainer
explainer = shap.LinearExplainer(model, X_transformed)
shap_values = explainer.shap_values(X_transformed)

# 5. Global Explainability (The "Why" for the whole model)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_transformed, show=False)
plt.title("Global Feature Importance (Logistic Regression)")
plt.tight_layout()
plt.savefig("./notebooks/shap_summary.png")
print("✅ Global summary plot saved to notebooks/.")

# 6. Local Explainability (Explanation for the first customer)
# We use a bar plot here as it is more stable with sparse-converted data
plt.figure(figsize=(8, 4))
shap.bar_plot(shap_values[0], max_display=10, show=False)
plt.title("Local Risk Drivers for Single Applicant")
plt.tight_layout()
plt.savefig("./notebooks/shap_local_explanation.png")
print("✅ Local explanation plot saved to notebooks/.")