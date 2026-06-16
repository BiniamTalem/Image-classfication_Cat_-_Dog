from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: str
    confidence: float
    probabilities: Optional[dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "dog",
                "confidence": 0.95,
                "probabilities": {"cat": 0.05, "dog": 0.95}
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    version: str = "1.0.0"