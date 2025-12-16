import pandas as pd
import numpy as np
import os
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

# Ensure data directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# Global variables
MLFLOW_EXPERIMENT_NAME = "Credit_Risk_Behavioral_Model"
TARGET_COLUMN = 'is_high_risk'
SNAPSHOT_DATE = pd.to_datetime('2019-02-05')

def load_data():
    """Loads raw transactions, assuming the file exists from previous steps."""
    file_path = "data/raw/data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at {file_path}. Please ensure Task 2 completed successfully.")
    df = pd.read_csv(file_path)
    return df

def calculate_rfm(df):
    """Calculates Recency, Frequency, and Monetary features."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # FIX 1: Remove timezone information (tz-aware vs tz-naive error)
    if df['TransactionStartTime'].dt.tz is not None:
        df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)

    # Aggregate transaction data by CustomerId
    customer_df = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (SNAPSHOT_DATE - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()

    # Apply simple clustering (k-means placeholder based on observations)
    # Cluster 1 (high Recency, low Frequency/Monetary) is typically high-risk.
    customer_df['Cluster'] = 0
    # Simple heuristic to identify high-risk cluster (Cluster 1)
    customer_df.loc[
        (customer_df['Recency'] > 30) &
        (customer_df['Frequency'] < 5) &
        (customer_df['Monetary'] < 20000),
        'Cluster'
    ] = 1
    # Simple heuristic to identify high-value cluster (Cluster 2)
    customer_df.loc[
        (customer_df['Frequency'] > 50) |
        (customer_df['Monetary'] > 500000),
        'Cluster'
    ] = 2

    # Define the target variable: 'is_high_risk'
    # We label Cluster 1 as high risk (True) and others as False
    customer_df[TARGET_COLUMN] = (customer_df['Cluster'] == 1).astype(int)

    return customer_df

def generate_customer_aggregates(df):
    """Generates additional customer-level transaction aggregates."""
    agg_df = df.groupby('CustomerId')['Amount'].agg(
        total_amount='sum',
        avg_amount='mean',
        std_amount='std',
        transaction_count='count'
    ).reset_index()

    # Fill NaNs from std (customers with one transaction)
    agg_df['std_amount'] = agg_df['std_amount'].fillna(0)

    # Get the last transaction values for categorical features
    last_tx_df = df.sort_values('TransactionStartTime', ascending=False).drop_duplicates('CustomerId', keep='first')

    # Select categorical features from the last transaction
    cat_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    last_tx_df = last_tx_df[['CustomerId'] + cat_features]

    # Merge all features
    df_merged = agg_df.merge(last_tx_df, on='CustomerId', how='left')
    return df_merged

def setup_preprocessor(numerical_indices, categorical_indices):
    """Sets up the standard preprocessing pipeline using feature indices."""

    numerical_transformer = Pipeline(steps=[
        # FIX (Previous step): Added with_mean=False to handle potential sparse data interaction
        ('scaler', StandardScaler(with_mean=False)) 
    ])

    categorical_transformer = Pipeline(steps=[
        # FIX (New): Force OneHotEncoder to output a dense array, making the entire ColumnTransformer output dense.
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_indices),
            ('cat', categorical_transformer, categorical_indices)
        ],
        remainder='drop'
        # By setting sparse_output=False in OHE, the entire output of CT will be dense (NumPy array)
    )

    try:
        from sklearn.set_output import set_output as sk_set_output
        sk_set_output(preprocessor, transform="pandas")
        print("INFO: ColumnTransformer set to output Pandas DataFrames.")
    except (NameError, ImportError, AttributeError):
        print("Warning: Could not set ColumnTransformer output to Pandas. Falling back to default output.")

    return preprocessor

def evaluate_model(model, X_test, y_test):
    """Calculates and returns metrics for the given model."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics

def train_and_log(model_name, estimator, params, X, X_test, y_train, y_test, preprocessor):
    """Trains a model, evaluates it, and logs everything to MLflow."""

    with mlflow.start_run(run_name=model_name) as run:

        # 1. Create the full pipeline (PreProcessor + Classifier)
        # Note: 'estimator' here is either the LR or the best LGBM *Pipeline* from GridSearchCV
        if isinstance(estimator, Pipeline):
            # This handles the LightGBM case where the best estimator from GS is already a Pipeline
            full_pipeline = estimator
        else:
            # This handles the Logistic Regression case where we build the Pipeline
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', estimator)
            ])


        # 2. Fit the model
        full_pipeline.fit(X, y_train) 

        # 3. Evaluate the model
        metrics = evaluate_model(full_pipeline, X_test, y_test)

        # 4. Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Handle signature inference for the dense NumPy array input
        # We transform the data using the preprocessor outside the pipeline to get the column count for signature
        X_transformed = full_pipeline.named_steps['preprocessor'].transform(X_test) 
        signature = infer_signature(X_transformed, full_pipeline.predict(X_test))

        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=signature, 
            registered_model_name=model_name
        )

        print(f"Metrics for {model_name}: {metrics}")

        return metrics, full_pipeline, run.info.run_id

def main_train():
    """Orchestrates data processing, training, hyperparameter tuning, and MLflow logging."""

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("1. Loading and processing raw data (Tasks 3 & 4)...")
    raw_df = load_data()
    rfm_df = calculate_rfm(raw_df.copy())
    features_df = generate_customer_aggregates(raw_df.copy())

    df = features_df.merge(rfm_df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Cluster', TARGET_COLUMN]], on='CustomerId')

    # Display cluster means and save processed data
    print("\n--- RFM Cluster Analysis (Mean Scores) ---")
    print(df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())
    print("\nCluster 1 identified as High-Risk.")

    # Save processed features
    df.to_csv("data/processed/customer_features.csv", index=False)
    print(f"Processed customer dataset created with {df['CustomerId'].nunique()} unique customers, RFM, and '{TARGET_COLUMN}' target.")
    print("Processed features saved to data/processed/customer_features.csv")

    # 2. Setup training
    X = df.drop(columns=['CustomerId', 'Cluster', TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # --- FEATURE LISTS AND INDICES SETUP (Strict Ordering Fix) ---
    numerical_features = ['Recency', 'Frequency', 'Monetary', 'total_amount', 'avg_amount', 'std_amount', 'transaction_count']
    categorical_features = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # Define the EXACT order of the features in the input NumPy array
    all_features_in_order = numerical_features + categorical_features
    
    # Map the indices based on this explicit order 
    numerical_indices = list(range(len(numerical_features)))
    categorical_indices = list(range(len(numerical_features), len(all_features_in_order)))
    # --- END FEATURE LISTS AND INDICES SETUP ---

    # Pass indices to the preprocessor
    preprocessor = setup_preprocessor(numerical_indices, categorical_indices)
    
    # --- Convert to NumPy array with STRICT column ordering ---
    # This is the data used for the LR train and Grid Search
    X_train_np = X_train[all_features_in_order].values
    X_test_np = X_test[all_features_in_order].values

    # --- 3. Baseline Model (Logistic Regression) ---
    print("\nTraining model: LR_Baseline...")
    lr_estimator = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    lr_params = {'solver': 'liblinear', 'class_weight': 'balanced'}

    # Pass NumPy arrays to train_and_log
    lr_metrics, lr_pipeline, lr_run_id = train_and_log(
        "LR_Baseline", lr_estimator, lr_params, X_train_np, X_test_np, y_train, y_test, preprocessor
    )

    # --- 4. Advanced Model (LightGBM with Grid Search) ---
    print("\nStarting Grid Search for LightGBM (may take a moment)...")

    lgbm_base_estimator = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)

    # Pipeline for Grid Search 
    lgbm_grid_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgbm_base_estimator)
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [10, 20]
    }

    grid_search = GridSearchCV(
        lgbm_grid_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0
    )

    grid_search.fit(X_train_np, y_train)

    # Extract best estimator and parameters
    # The best_estimator_ is already a full Pipeline
    lgbm_best = grid_search.best_estimator_
    lgbm_params = {'model_type': 'LightGBM_Tuned', **grid_search.best_params_}

    print("\nTraining model: LightGBM_Tuned...")
    
    # Pass the full best pipeline to train_and_log 
    lgbm_metrics, lgbm_pipeline, lgbm_run_id = train_and_log(
        "LightGBM_Tuned", lgbm_best, lgbm_params, X_train_np, X_test_np, y_train, y_test, preprocessor
    )

    # --- 5. Model Comparison and Registration ---
    print("\n--- Model Comparison ---")
    print(f"LR_Baseline ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"LightGBM_Tuned ROC-AUC: {lgbm_metrics['roc_auc']:.4f}")

    if lgbm_metrics['roc_auc'] > lr_metrics['roc_auc']:
        best_model_run_id = lgbm_run_id
        best_model_name = "LightGBM_Tuned"
    else:
        best_model_run_id = lr_run_id
        best_model_name = "LR_Baseline"

    # Register the best model
    model_uri = f"runs:/{best_model_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="Best_Credit_Risk_Model")

    print(f"\nSuccessfully trained, evaluated, and logged two models.")
    print(f"The best model is {best_model_name}, registered as 'Best_Credit_Risk_Model' in MLflow.")


if __name__ == "__main__":
    main_train()