from pydantic import BaseModel, Field
from typing import List

# Define the exact features used by the model (NOW 11)
class CustomerFeatures(BaseModel):
    Recency: int = Field(..., description="Days since last transaction.")
    Frequency: int = Field(..., description="Total number of transactions.")
    Monetary: float = Field(..., description="Total monetary value of transactions.")
    transaction_count: int
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int

    class Config:
        json_schema_extra = {
            "example": {
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
        }

# Define the response structure
class PredictionResponse(BaseModel):
    risk_prediction: str = Field(..., description="Predicted risk label (LOW RISK or HIGH RISK).")
    risk_probability: float = Field(..., description="Probability of being high risk (0.0 to 1.0).")