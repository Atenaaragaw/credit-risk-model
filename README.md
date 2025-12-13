# Credit Risk Probability Model for Alternative Data

## Project Overview

This project implements an end-to-end credit risk model for Bati Bank's new Buy-Now-Pay-Later (BNPL) service, leveraging alternative data (eCommerce transaction history) to assign a risk probability score to customers.

---

## 1. Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

* **Basel II and Interpretation:** The Accord requires banks to quantify and manage their risk exposure (Credit, Operational, Market). This regulatory scrutiny demands that models are **highly interpretable, auditable, and well-documented**. A "black box" model is unacceptable because regulators must be able to understand *why* a decision was made and *how* the risk parameters were calculated.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

* **Necessity of a Proxy:** We must use a proxy (like an RFM-based "disengagement" score) because the raw data lacks a clear, mandated label for **default**. The proxy links observable customer behavior (e.g., low frequency/monetary value) to the theoretical concept of **high credit risk**.
* **Business Risks of Proxy:** The major risk is **Proxy Misalignment**. This leads to **False Negatives** (lending to a defaulter) and **False Positives** (denying credit to a reliable customer), resulting in financial losses or lost revenue.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Model Type | Advantage in Regulated Finance | Trade-Off / Disadvantage |
| :--- | :--- | :--- |
| **Simple (e.g., LogReg + WoE)** | **High Interpretability:** Easy to explain to regulators and customers. Essential for compliance. | **Lower Predictive Power:** May not capture complex non-linear relationships. |
| **Complex (e.g., Gradient Boosting)** | **High Predictive Power:** Excellent for capturing complex patterns, leading to more accurate risk prediction. | **Low Interpretability:** The model's decision process is harder to trace. Requires extra tools (SHAP/LIME) for justification. |

### Step 6: Install Dependencies

**Your Action:** With the virtual environment active, install all the listed packages.

```bash
pip install -r requirements.txt