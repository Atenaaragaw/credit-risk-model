import pandas as pd
import numpy as np
import mlflow.sklearn # ⬅️ We import the specific MLflow flavor loader
from sklearn.exceptions import NotFittedError

# Define the model name used for registration in train.py
MODEL_NAME = "Best_Credit_Risk_Model" 

# --- Two samples of new customer data for prediction ---
# The features must be in the exact order the model was trained on:
FEATURE_NAMES = [
    'Recency', 'Frequency', 'Monetary', 'total_amount', 'avg_amount', 'std_amount', 
    'transaction_count', 'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
    'ProductCategory', 'ChannelId', 'PricingStrategy'
]

NEW_CUSTOMER_DATA = [
    # Customer A (Expected LOW RISK: Recent, High Frequency/Monetary)
    [
        10,            # Recency (Low)
        25,            # Frequency (High)
        150000.0,      # Monetary (High)
        150000.0,      # total_amount
        6000.0,        # avg_amount
        2000.0,        # std_amount
        25,            # transaction_count
        'TZS',         
        'Tanzania',    
        'ProviderA',   
        'ProductX',    
        'financial_services', 
        'Web',         
        1              
    ],
    # Customer B (Expected HIGH RISK: Dormant, Low Frequency/Monetary - fits Cluster 1 profile)
    [
        90,            # Recency (High - Dormant)
        1,             # Frequency (Low)
        1500.0,        # Monetary (Very Low)
        1500.0,        # total_amount
        1500.0,        # avg_amount
        0.0,           # std_amount
        1,             # transaction_count
        'TZS',         
        'Tanzania',    
        'ProviderB',   
        'ProductY',    
        'financial_services', 
        'App',         
        4              
    ]
]

X_new = pd.DataFrame(NEW_CUSTOMER_DATA, columns=FEATURE_NAMES)

def make_predictions(X_data: pd.DataFrame):
    """
    Loads the best model from MLflow Model Registry and makes predictions.
    """
    try:
        print(f"Loading the latest registered version of model: '{MODEL_NAME}'...")
        
        # Load the model by its registered name
        model_uri = f"models:/{MODEL_NAME}/latest"
        # ⬅️ FIXED: Use mlflow.sklearn.load_model to get the native Pipeline object
        loaded_model = mlflow.sklearn.load_model(model_uri) 
        
        print("Model loaded successfully.")
        
        # --- Prepare Input Data: Convert DataFrame to NumPy array to match the training input format ---
        # This is essential to match the input the model expects from the training run.
        X_data_ordered = X_data[FEATURE_NAMES]
        X_data_np = X_data_ordered.values
        # ---------------------------------------------------------------------------------------------
        
        # The loaded model is the full scikit-learn pipeline, preserving predict_proba
        predictions = loaded_model.predict(X_data_np)
        probabilities = loaded_model.predict_proba(X_data_np)[:, 1] # Now this should work!

        results = X_data.copy()
        results['Risk_Prediction'] = np.where(predictions == 1, 'HIGH RISK', 'LOW RISK')
        results['Risk_Probability'] = probabilities
        
        print("\n--- INFERENCE RESULTS ---")
        print(results[['Recency', 'Frequency', 'Monetary', 'Risk_Prediction', 'Risk_Probability']])
        
    except NotFittedError as e:
        print(f"Error: The model or one of its steps is not fitted. Ensure train.py ran successfully. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        print(f"Hint: Check that the MLflow tracking server is running or if the model '{MODEL_NAME}' is registered.")

if __name__ == "__main__":
    make_predictions(X_new)