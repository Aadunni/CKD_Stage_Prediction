from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from enum import Enum

class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"

class CKDStageEnum(int, Enum):
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    STAGE_4 = 4
    STAGE_5 = 5

class PatientData(BaseModel):
    """Input schema for patient data"""
    sodium: float = Field(..., gt=0, description="Sodium level in mmol/L")
    potassium: float = Field(..., gt=0, description="Potassium level in mmol/L")
    glucose: float = Field(..., gt=0, description="Glucose level in mmol/L")
    urea: float = Field(..., gt=0, description="Urea level in mmol/L")
    body_temperature: float = Field(..., gt=0, description="Body temperature in Celsius")
    sbp: float = Field(..., gt=0, description="Systolic Blood Pressure")
    dbp: float = Field(..., gt=0, description="Diastolic Blood Pressure")
    weight: float = Field(..., gt=0, description="Weight in kg")
    age: int = Field(..., ge=0, description="Age of patient")
    gender: GenderEnum = Field(..., description="Gender of patient")
    creatinine: float = Field(..., gt=0, description="Creatinine level in mg/dl")

    @validator('sbp')
    def validate_sbp(cls, v, values):
        if 'dbp' in values and v <= values['dbp']:
            raise ValueError('SBP must be greater than DBP')
        return v

    class Config:
        schema_extra = {
            "example": {
                "sodium": 140.0,
                "potassium": 4.0,
                "glucose": 5.5,
                "urea": 7.0,
                "body_temperature": 36.5,
                "sbp": 120,
                "dbp": 80,
                "weight": 70.0,
                "age": 45,
                "gender": "male",
                "creatinine": 1.2
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predicted_stage: int = Field(..., description="Predicted CKD stage (0-5)")
    predicted_stage_name: str = Field(..., description="Stage name description")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    risk_level: str = Field(..., description="Risk assessment")
    recommendations: list = Field(..., description="Clinical recommendations")

class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    api_version: str
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    timestamp: str