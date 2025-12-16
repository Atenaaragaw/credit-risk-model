import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
RAW_DATA_PATH = 'data/raw/data.csv'
SNAPSHOT_DATE = pd.to_datetime('2019-02-05') 


# --- 2. CORE UTILITY FUNCTIONS (Task 3 & 4) ---

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw transactional data and performs initial type correction."""
    df = pd.read_csv(path)
    
    # 1. Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    # FIX for Timezone Awareness Conflict: Remove timezone info to align with SNAPSHOT_DATE
    df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)
    
    # 2. Drop redundant column
    df.drop(columns=['Value'], inplace=True) 
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts date/time components as features."""
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df

def generate_customer_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactional data to the customer level, creating statistical features."""
    customer_features = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std'),
        transaction_count=('TransactionId', 'count')
    ).reset_index()
    
    customer_features['std_amount'] = customer_features['std_amount'].fillna(0)
    
    return customer_features


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values for each customer."""
    
    # 1. Recency: Time since last transaction
    recency_df = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency_df.columns = ['CustomerId', 'LastTransactionDate']
    recency_df['Recency'] = (SNAPSHOT_DATE - recency_df['LastTransactionDate']).dt.days

    # 2. Frequency & Monetary (uses ABSOLUTE amount for total flow of funds)
    rfm_df = df.groupby('CustomerId').agg(
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', lambda x: x.abs().sum())
    ).reset_index()
    
    rfm_df = pd.merge(rfm_df, recency_df[['CustomerId', 'Recency']], on='CustomerId', how='inner')
    
    return rfm_df[['CustomerId', 'Recency', 'Frequency', 'Monetary']]


def create_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the 'is_high_risk' binary target variable using K-Means clustering
    on the RFM features. 
    """
    # Use Log-transformed features for clustering (to handle skew/outliers)
    df['LogMonetary'] = np.log1p(df['Monetary'])
    df['LogFrequency'] = np.log1p(df['Frequency'])
    
    clustering_features = ['Recency', 'LogFrequency', 'LogMonetary']
    
    # Standardization (mean=0, std=1)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df[clustering_features])
    
    # --- K-Means Clustering (K=3 mandated) ---
    K = 3 
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # --- Defining the High-Risk Label ---
    cluster_means = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print("\n--- RFM Cluster Analysis (Mean Scores) ---")
    print(cluster_means)
    
    # Identify the 'High-Risk' cluster (High R, Low F, Low M) using a risk ranking heuristic:
    risk_score = (cluster_means['Recency'].rank(ascending=False) 
                  - cluster_means['Frequency'].rank(ascending=True) 
                  - cluster_means['Monetary'].rank(ascending=True))
                  
    high_risk_cluster = risk_score.idxmax()
    
    print(f"\nCluster {high_risk_cluster} identified as High-Risk.")

    # Assign binary target variable
    df['is_high_risk'] = (df['Cluster'] == high_risk_cluster).astype(int)
    
    # Drop intermediate columns
    df.drop(columns=['Cluster', 'LogMonetary', 'LogFrequency'], inplace=True, errors='ignore')
    
    return df


# --- 3. MAIN ORCHESTRATION FUNCTION ---

def process_raw_data() -> pd.DataFrame:
    """
    Main function to orchestrate the raw data processing pipeline,
    including feature engineering, RFM calculation, and target variable creation.
    """
    df = load_data()
    df = create_time_features(df)
    
    # 1. Generate customer-level aggregate features
    customer_agg = generate_customer_aggregates(df)
    
    # 2. Calculate RFM features
    rfm_df = calculate_rfm(df)
    
    # 3. Merge aggregates and RFM features
    processed_df = pd.merge(customer_agg, rfm_df[['CustomerId', 'Recency', 'Frequency', 'Monetary']], on='CustomerId', how='inner')
    
    # 4. Create the Proxy Target Variable (is_high_risk)
    processed_df = create_risk_target(processed_df)
    
    # 5. Get latest transaction information (categorical features for model input)
    latest_transactions = df.sort_values('TransactionStartTime', ascending=False).drop_duplicates(subset=['CustomerId'])
    latest_info = latest_transactions[[
        'CustomerId', 'CurrencyCode', 'CountryCode', 'ProviderId', 
        'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]].copy()
    
    # Merge latest info 
    final_processed_df = pd.merge(processed_df, latest_info, on='CustomerId', how='left')
    
    print(f"Processed customer dataset created with {final_processed_df.shape[0]} unique customers, RFM, and 'is_high_risk' target.")
    return final_processed_df


# --- 4. EXECUTION AND VALIDATION ---
if __name__ == '__main__':
    # Simple test run
    processed_df = process_raw_data()
    print(processed_df.head())
    print("\n--- Target Variable Distribution ---")
    print(processed_df['is_high_risk'].value_counts(normalize=True))