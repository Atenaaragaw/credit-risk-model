import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import the data processing function from our module
# NOTE: This assumes src/data_processing.py is accessible via the path
try:
    from data_processing import process_raw_data
except ImportError:
    # If running from a different directory, try relative import
    from src.data_processing import process_raw_data


# --- CONFIGURATION ---
RANDOM_SEED = 42
MODEL_FILE = 'model/credit_risk_model.joblib'
DATA_FILE = 'data/processed/customer_features.csv'

def setup_preprocessor(numerical_features, categorical_features):
    """
    Sets up the preprocessing pipeline for numerical and categorical features 
    as required by Task 3 (Scaling, Encoding, Missing Value Handling).
    """
    # Pipeline for Numerical Features (Standardization/Scaling)
    numerical_transformer = Pipeline(steps=[
        # Note: Missing values are handled at the source (clean data), but SimpleImputer can be added here if needed
        ('scaler', StandardScaler())
    ])

    # Pipeline for Categorical Features (One-Hot Encoding, meeting Task 3 requirement)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop all other columns (like CustomerId, Recency, Frequency, Monetary)
    )
    return preprocessor

def train_and_evaluate_model():
    """
    Orchestrates the model training, evaluation, and saving process.
    """
    # 1. Load and Process Data (from Task 4)
    print("1. Loading and processing raw data (Task 3 & 4 complete)...")
    df = process_raw_data()
    
    # Save processed features for potential future use (e.g., in API)
    df.to_csv(DATA_FILE, index=False)
    print(f"Processed features saved to {DATA_FILE}")

    # 2. Define Features and Target
    X = df.drop(columns=['CustomerId', 'is_high_risk'])
    y = df['is_high_risk']
    
    # IMPORTANT: Drop the RFM base columns as they were used to create the target via clustering.
    # We will only use the aggregate features (total_amount, etc.) and categorical features for the model.
    X = X.drop(columns=['Recency', 'Frequency', 'Monetary'])

    # Identify final feature types for preprocessing
    numerical_features = ['total_amount', 'avg_amount', 'std_amount', 'transaction_count']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 
                            'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # 3. Split Data (Stratified split is crucial due to target imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 4. Setup Preprocessor (Handles Scaling and Encoding)
    preprocessor = setup_preprocessor(numerical_features, categorical_features)
    
    # 5. Define Model (Logistic Regression as a robust baseline)
    # class_weight='balanced' helps handle the 68/32 target split
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_SEED)

    # 6. Create Full Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    print("2. Training Logistic Regression model...")

    # 7. Train Model
    full_pipeline.fit(X_train, y_train)
    
    # 8. Predict and Evaluate
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    # Use a dictionary for clear evaluation results
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    print("\n3. Model Evaluation Results (on Test Set):")
    for metric, value in metrics.items():
        print(f"   {metric:<10}: {value:.4f}")

    # 9. Save Model
    joblib.dump(full_pipeline, MODEL_FILE)
    print(f"\n4. Model saved successfully to {MODEL_FILE}")


if __name__ == '__main__':
    # Ensure necessary directories exist
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    train_and_evaluate_model()