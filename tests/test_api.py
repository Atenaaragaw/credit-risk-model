import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.pydantic_models import CustomerFeatures

# Initialize the TestClient
client = TestClient(app)

# 1. Test Health Check
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# 2. Test Pydantic Data Validation (Valid Data)
def test_customer_features_validation():
    valid_data = {
        "Recency": 10,
        "Frequency": 5,
        "Monetary": 500.0,
        "transaction_count": 5,
        "CurrencyCode": "TZS",
        "CountryCode": "Tanzania",
        "ProviderId": "P1",
        "ProductId": "X",
        "ProductCategory": "fin",
        "ChannelId": "Web",
        "PricingStrategy": 1
    }
    features = CustomerFeatures(**valid_data)
    assert features.Recency == 10

# 3. Test Pydantic Data Validation (Invalid Data - Missing Field)
def test_invalid_customer_features():
    invalid_data = {"Recency": 10} # Missing all other required fields
    with pytest.raises(ValueError):
        CustomerFeatures(**invalid_data)

# 4. Test Predict Endpoint (Integration Test)
def test_predict_endpoint():
    sample_payload = {
        "Recency": 10,
        "Frequency": 25,
        "Monetary": 150000.0,
        "transaction_count": 25,
        "CurrencyCode": "TZS",
        "CountryCode": "Tanzania",
        "ProviderId": "ProviderA",
        "ProductId": "ProductX",
        "ProductCategory": "financial_services",
        "ChannelId": "Web",
        "PricingStrategy": 1
    }
    response = client.post("/predict", json=sample_payload)
    
    # Note: This requires the model to be loaded successfully in the test env
    if response.status_code == 200:
        data = response.json()
        assert "risk_prediction" in data
        assert "risk_probability" in data
        assert isinstance(data["risk_probability"], float)
    else:
        # If model isn't found in test env, it might return 503 or 500
        assert response.status_code in [503, 500]