from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Dict

# Import local modules
from models.schemas import PatientData, PredictionResponse, HealthCheckResponse, ErrorResponse
from core.model_manager import ModelManager
from core.exceptions import ModelNotLoadedException, PredictionException
from utils.helpers import setup_logging, get_current_timestamp, format_probabilities

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CKD Prediction API",
    description="API for predicting Chronic Kidney Disease stages using Machine Learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=get_current_timestamp()
        ).dict()
    )

# API Routes
@app.get("/", response_model=Dict[str, str])
async def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to the CKD Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        api_version="1.0.0",
        timestamp=get_current_timestamp()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_ckd(patient_data: PatientData):
    """Predict CKD stage for a patient"""
    try:
        # Make prediction
        predicted_stage, probabilities = model_manager.predict(patient_data)
        
        # Get stage description
        stage_name = model_manager.stage_descriptions.get(predicted_stage, "Unknown")
        
        # Get confidence (probability of predicted class)
        confidence = float(probabilities[predicted_stage]) if len(probabilities) > predicted_stage else 0.0
        
        # Format probabilities
        class_labels = list(range(len(probabilities)))
        formatted_probs = format_probabilities(probabilities, class_labels)
        
        # Get risk level and recommendations
        risk_level = model_manager.get_risk_level(predicted_stage)
        recommendations = model_manager.get_recommendations(predicted_stage)
        
        return PredictionResponse(
            predicted_stage=predicted_stage,
            predicted_stage_name=stage_name,
            confidence=confidence,
            probabilities=formatted_probs,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise PredictionException(str(e))

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "model_type": "AdaBoost Classifier",
        "features": model_manager.feature_columns,
        "classes": list(model_manager.stage_descriptions.keys()),
        "class_descriptions": model_manager.stage_descriptions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)