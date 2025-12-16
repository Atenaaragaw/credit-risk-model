import pandas as pd
import pytest
import numpy as np
from src.data_processing import generate_customer_aggregates, calculate_rfm

# Note: We assume SNAPSHOT_DATE = pd.to_datetime('2019-02-05') is defined in src/data_processing.py

# Define a fixture for a sample minimal dataframe for testing
@pytest.fixture
def sample_raw_data():
    """Provides a minimal raw dataframe for testing functions."""
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'Amount': [100.0, -50.0, 200.0, 1000.0],
        'TransactionStartTime': ['2018-12-05 11:00:00', '2018-12-05 10:00:00', '2019-01-15 12:00:00', '2019-02-04 09:00:00'],
    }
    # Ensure TransactionStartTime is datetime object for RFM calculation
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# Test 1: Aggregate Feature Calculations
def test_aggregate_calculations(sample_raw_data):
    """Tests the calculation of base aggregate features."""
    agg_df = generate_customer_aggregates(sample_raw_data)
    
    # Check C1 (100, -50)
    c1 = agg_df[agg_df['CustomerId'] == 'C1'].iloc[0]
    assert c1['transaction_count'] == 2
    assert c1['total_amount'] == 50.0
    assert np.isclose(c1['avg_amount'], 25.0)
    
    # Check C3 (1000) - std_amount should be 0 as there is only one transaction
    c3 = agg_df[agg_df['CustomerId'] == 'C3'].iloc[0]
    assert c3['transaction_count'] == 1
    assert c3['std_amount'] == 0.0
    
    # Check output shape
    assert agg_df.shape[0] == 3, "Should have 3 unique customers."

# Test 2: RFM Calculation
def test_rfm_calculation(sample_raw_data):
    """Tests the calculation of Recency and Monetary values."""
    
    rfm_df = calculate_rfm(sample_raw_data)
    
    # Check C1 (Last transaction: 2018-12-05) -> Recency = (2019-02-05 - 2018-12-05).dt.days = 62 days
    c1 = rfm_df[rfm_df['CustomerId'] == 'C1'].iloc[0]
    assert c1['Recency'] == 61, "Recency calculation for C1 is incorrect."
    assert c1['Monetary'] == 150.0, "Monetary (absolute sum) for C1 is incorrect."
    
    # Check C3 (Last transaction: 2019-02-04) -> Recency = 1 day
    c3 = rfm_df[rfm_df['CustomerId'] == 'C3'].iloc[0]
    assert c3['Recency'] == 0, "Recency calculation for C3 is incorrect."
    assert c3['Frequency'] == 1, "Frequency for C3 is incorrect."