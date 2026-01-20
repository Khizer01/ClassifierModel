from pydantic import BaseModel, Field
from typing import List

class ClassificationRequest(BaseModel):
    keyword: str = Field(..., description="Brand keyword to search for and classify ads")

class AdClassificationResult(BaseModel):
    ad_id: str = Field(..., description="Unique identifier for the ad")
    is_relevant: bool = Field(..., description="Whether the ad is relevant to the keyword based on promotional intent")
    theme: str = Field(..., description="Primary theme of the ad")

class ClassificationResponse(BaseModel):
    results: List[AdClassificationResult] = Field(..., description="List of classification results")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
