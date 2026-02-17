from pydantic import BaseModel, Field, ConfigDict
from typing import List

# Final Refactored Model for Finance Sector Reliability
class CustomerFeatures(BaseModel):
    # Use ConfigDict for Pydantic V2 compatibility
    model_config = ConfigDict(
        json_schema_extra={
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
    )

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

class PredictionResponse(BaseModel):
    risk_prediction: str
    risk_probability: float