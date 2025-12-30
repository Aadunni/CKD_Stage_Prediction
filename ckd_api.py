#!/usr/bin/env python3
"""
CKD Prediction API using Random Forest Model
FastAPI server for Chronic Kidney Disease prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from google import genai

# Load environment variables from .env file
load_dotenv()

# Global variables for model components  
model = None
scaler = None  # Not used for Random Forest, but kept for compatibility
feature_names = None

gemini_client = None

# LLM Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# GEMINI_API_URL is no longer used for the SDK, but kept for context.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def load_model_components():
    """Load the trained model and feature names"""
    global model, scaler, feature_names
    
    try:
        # Find the latest model files (with timestamp)
        models_dir = Path('models')
        
        # Look for the latest Random Forest model
        rf_model_files = list(models_dir.glob('best_model_random_forest_*.pkl'))
        feature_files = list(models_dir.glob('feature_names_*.pkl'))
        
        if not rf_model_files:
            # Fallback to old naming convention
            model = joblib.load('models/random_forest_model.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            print("📁 Loaded legacy model files")
        else:
            # Use the latest timestamped files
            latest_model = max(rf_model_files, key=lambda x: x.name)
            latest_features = max(feature_files, key=lambda x: x.name)
            
            model = joblib.load(latest_model)
            feature_names = joblib.load(latest_features)
            print(f"📁 Loaded latest model: {latest_model.name}")
            print(f"📁 Loaded latest features: {latest_features.name}")
        
        # Random Forest doesn't need scaling - explicitly set to None
        scaler = None
        print("ℹ️  Random Forest model - no scaling required")
        
        print("✅ Model components loaded successfully!")
        print(f"🏷️  Model type: {type(model).__name__}")
        print(f"📋 Expected features ({len(feature_names)}): {feature_names}")
        
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model_components()
    
    # >>> GEMINI CLIENT INITIALIZATION <<<
    global gemini_client
    print(f"🔍 Debug: GEMINI_API_KEY present: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"🔍 Debug: GEMINI_API_KEY length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}")
    
    if GEMINI_API_KEY:
        try:
            print("🔧 Attempting to create genai.Client...")
            # The client gets the API key from the environment variable `GEMINI_API_KEY` automatically
            gemini_client = genai.Client()
            print("✅ Gemini Client initialized using new SDK.")
            
            # Test the client with a simple call
            print("🔧 Testing client with simple prompt...")
            test_response = gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents="Hello, respond with 'API Working'"
            )
            print(f"✅ Test successful: {test_response.text[:50]}...")
            
        except ImportError as e:
            print(f"❌ Import Error with google.genai: {e}")
            print("💡 Suggestion: Install with 'pip install google-genai'")
            gemini_client = None
        except Exception as e:
            print(f"❌ Error initializing Gemini Client: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            gemini_client = None # Ensure it is explicitly None if initialization fails
    else:
        print("⚠️ GEMINI_API_KEY not found in environment variables")
        gemini_client = None
    
    print(f"🔍 Final gemini_client state: {'Initialized' if gemini_client else 'None'}")
    # >>> END GEMINI CLIENT INITIALIZATION <<<
    
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="CKD Prediction API",
    description="Chronic Kidney Disease Stage 5 Prediction using Random Forest Model",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response (KEPT AS IS)
class PatientData(BaseModel):
    """Single patient data for CKD prediction"""
    age: float = Field(..., description="Age of Patient (year)", ge=0, le=120)
    weight: float = Field(..., description="Weight (kg)", ge=20, le=200)
    creatinine: float = Field(..., description="Creatinine (Umol/L)", ge=0)
    sodium: float = Field(..., description="Sodium (mmol/L)", ge=120, le=160)
    potassium: float = Field(..., description="Potassium (mmol/L)", ge=2.0, le=8.0)
    glucose: float = Field(..., description="Glucose (mmol/L)", ge=2.0, le=30.0)
    urea: float = Field(..., description="Urea (mmol/L)", ge=1.0, le=50.0)
    gender_male: int = Field(..., description="Gender_Male (1 for Male, 0 for Female)", ge=0, le=1)

class BatchPatientData(BaseModel):
    """Batch of patient data for multiple predictions"""
    patients: List[PatientData] = Field(..., description="List of patient data")

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    patient_id: int
    prediction: int  # 0 = No CKD, 1 = CKD Stage 5
    status: str  # "CKD Detected" or "No CKD Detected"
    probability: float  # Probability of CKD Stage 5
    risk_level: str  # "Low", "Medium", "High"
    confidence: float  # Model confidence (max probability)
    message: str  # Clear interpretation message

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class LLMAssessmentRequest(BaseModel):
    """Request model for LLM medical assessment"""
    patient_data: PatientData
    model_prediction: PredictionResponse
    additional_context: Optional[str] = Field(None, description="Additional medical history or context")

class LLMAssessmentResponse(BaseModel):
    """Response model for LLM medical assessment"""
    clinical_interpretation: str
    lab_values_analysis: str
    model_validation: str
    risk_factors: List[str]
    recommendations: List[str]
    confidence_in_model: str
    additional_tests_suggested: List[str]
    summary: str

def prepare_features(patient_data: PatientData) -> np.ndarray:
    """Convert patient data to model features"""
    
    # Create feature vector in correct order matching expected features:
    # ['Age of Patient (year)', 'Weight (kg)', 'Creatinine (Umol/L)', 
    #  'Sodium (mmol/L)', 'Potassium (mmol/L)', 'Glucose (mmol/L)', 
    #  'Urea (mmol/L)', 'Gender_Male']
    features = [
        patient_data.age,
        patient_data.weight, 
        patient_data.creatinine,
        patient_data.sodium,
        patient_data.potassium,
        patient_data.glucose,
        patient_data.urea,
        patient_data.gender_male
    ]
    
    return np.array(features).reshape(1, -1)

def get_risk_level(probability: float) -> str:
    """Determine risk level based on prediction probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium" 
    else:
        return "High"

def get_prediction_message(prediction: int, probability: float, risk_level: str) -> tuple:
    """Get status and message based on prediction"""
    if prediction == 1:
        status = "CKD Detected"
        message = f"⚠️ Chronic Kidney Disease Stage 5 detected with {probability:.1%} probability. Risk level: {risk_level}. Immediate medical consultation recommended."
    else:
        status = "No CKD Detected"
        message = f"✅ No Chronic Kidney Disease detected. CKD probability: {probability:.1%}. Risk level: {risk_level}. Continue regular health monitoring."
    
    return status, message

# >>> NEW SYNCHRONOUS SDK CALL FUNCTION <<<
def call_gemini_sdk_sync(prompt: str) -> str:
    """Synchronous function to call the Gemini API using the SDK."""
    global gemini_client
    
    print(f"🔍 Debug: call_gemini_sdk_sync called")
    print(f"🔍 Debug: gemini_client state: {'Available' if gemini_client else 'None'}")
    
    if gemini_client is None:
        print("❌ Debug: Gemini client is None - returning 503")
        raise HTTPException(
            status_code=503,
            detail="Gemini Client not initialized. API key might be missing or invalid."
        )
        
    try:
        print(f"🔧 Debug: Attempting to generate content with prompt length: {len(prompt)}")
        # Generate content using the new genai.Client API
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        print(f"✅ Debug: Response received, text length: {len(response.text) if response.text else 0}")
        return response.text
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        # Re-raise as HTTPException for FastAPI
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

# >>> REPLACING call_gemini_api with ASYNC SDK WRAPPER <<<
async def call_gemini_api_sdk_async(prompt: str) -> str:
    """Call Gemini API using the official SDK, running the sync call in a thread."""
    
    if gemini_client is None:
        raise HTTPException(
            status_code=503, 
            detail="LLM service not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    # Use asyncio.to_thread to run the synchronous SDK call without blocking the event loop
    return await asyncio.to_thread(call_gemini_sdk_sync, prompt)


def create_medical_prompt(patient_data: PatientData, prediction_response: PredictionResponse, additional_context: str = None) -> str:

    """Create a comprehensive medical prompt for LLM assessment"""
    
    gender = "Male" if patient_data.gender_male == 1 else "Female"
    
    prompt = f"""
    You are an expert nephrologist reviewing a patient case and a machine learning model's CKD prediction. Please provide a comprehensive medical assessment.

    **PATIENT PROFILE:**
    - Age: {patient_data.age} years
    - Gender: {gender}
    - Weight: {patient_data.weight} kg
    
    **LABORATORY VALUES:**
    - Creatinine: {patient_data.creatinine} μmol/L (Normal: Male <120, Female <110)
    - Sodium: {patient_data.sodium} mmol/L (Normal: 135-145)
    - Potassium: {patient_data.potassium} mmol/L (Normal: 3.5-5.5)
    - Glucose: {patient_data.glucose} mmol/L (Normal: 3.9-6.1 fasting)
    - Urea: {patient_data.urea} mmol/L (Normal: 2.5-8.0)
    
    **ML MODEL PREDICTION:**
    - Prediction: {prediction_response.prediction} ({'CKD Stage 5' if prediction_response.prediction == 1 else 'No CKD'})
    - Status: {prediction_response.status}
    - Probability: {prediction_response.probability:.1%}
    - Risk Level: {prediction_response.risk_level}
    - Model Confidence: {prediction_response.confidence:.1%}
    
    **ADDITIONAL CONTEXT:** {additional_context if additional_context else "None provided"}
    
    Please provide a structured assessment covering:
    
    1. **CLINICAL_INTERPRETATION**: Your overall clinical interpretation of this case
    2. **LAB_VALUES_ANALYSIS**: Detailed analysis of each lab value and what they indicate
    3. **MODEL_VALIDATION**: Do you agree with the ML model's prediction? Why or why not?
    4. **RISK_FACTORS**: List key risk factors present in this case
    5. **RECOMMENDATIONS**: Specific clinical recommendations and next steps
    6. **CONFIDENCE_IN_MODEL**: Your confidence level in the model's assessment
    7. **ADDITIONAL_TESTS**: Suggested additional tests or investigations
    8. **SUMMARY**: Brief summary and urgency level
    
    Please format your response with clear section headers and be specific about clinical reasoning.
    """
    
    return prompt

def parse_llm_response(llm_text: str) -> LLMAssessmentResponse:
    # ... (KEPT AS IS) ...
    """Parse LLM response into structured format"""
    
    # Default values in case parsing fails
    default_response = {
        "clinical_interpretation": llm_text[:500] + "..." if len(llm_text) > 500 else llm_text,
        "lab_values_analysis": "Please refer to full clinical interpretation above.",
        "model_validation": "Assessment provided in clinical interpretation.",
        "risk_factors": ["Refer to clinical interpretation"],
        "recommendations": ["Consult with nephrologist for detailed assessment"],
        "confidence_in_model": "Moderate confidence - clinical judgment recommended",
        "additional_tests_suggested": ["Standard nephrology workup as clinically indicated"],
        "summary": "Comprehensive clinical assessment provided above."
    }
    
    try:
        # Simple parsing - look for section headers
        sections = {}
        current_section = None
        current_content = []
        
        for line in llm_text.split('\n'):
            line = line.strip()
            
            # Check for section headers
            if any(header in line.upper() for header in [
                'CLINICAL_INTERPRETATION', 'LAB_VALUES_ANALYSIS', 'MODEL_VALIDATION',
                'RISK_FACTORS', 'RECOMMENDATIONS', 'CONFIDENCE_IN_MODEL',
                'ADDITIONAL_TESTS', 'SUMMARY'
            ]):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                if 'CLINICAL_INTERPRETATION' in line.upper():
                    current_section = 'clinical_interpretation'
                elif 'LAB_VALUES_ANALYSIS' in line.upper():
                    current_section = 'lab_values_analysis'
                elif 'MODEL_VALIDATION' in line.upper():
                    current_section = 'model_validation'
                elif 'RISK_FACTORS' in line.upper():
                    current_section = 'risk_factors'
                elif 'RECOMMENDATIONS' in line.upper():
                    current_section = 'recommendations'
                elif 'CONFIDENCE_IN_MODEL' in line.upper():
                    current_section = 'confidence_in_model'
                elif 'ADDITIONAL_TESTS' in line.upper():
                    current_section = 'additional_tests_suggested'
                elif 'SUMMARY' in line.upper():
                    current_section = 'summary'
                
                current_content = []
            elif current_section and line:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract list items for risk_factors, recommendations, additional_tests
        def extract_list_items(text: str) -> List[str]:
            items = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    items.append(line[1:].strip())
                elif line and not items:  # If no bullets, treat each line as an item
                    items.append(line)
            return items if items else [text]
        
        # Build response
        response_data = {}
        for key, default in default_response.items():
            if key in sections:
                if key in ['risk_factors', 'recommendations', 'additional_tests_suggested']:
                    response_data[key] = extract_list_items(sections[key])
                else:
                    response_data[key] = sections[key]
            else:
                response_data[key] = default
        
        return LLMAssessmentResponse(**response_data)
        
    except Exception as e:
        # Fallback to default response if parsing fails
        default_response['clinical_interpretation'] = f"Raw LLM Response:\n{llm_text}"
        return LLMAssessmentResponse(**default_response)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "CKD Prediction API is running!",
        "model": "Random Forest Classifier",
        "features": len(feature_names) if feature_names else 0,
        "status": "healthy"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "features": feature_names,
        "feature_count": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(patient: PatientData):
    """Predict CKD for a single patient"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(patient)
        
        # Random Forest works with raw features - no scaling needed
        # Make prediction directly with raw features
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get probability of CKD Stage 5 (class 1)
        ckd_probability = probabilities[1]
        confidence = max(probabilities)
        risk_level = get_risk_level(ckd_probability)
        
        # Get clear status and message
        status, message = get_prediction_message(int(prediction), ckd_probability, risk_level)
        
        return PredictionResponse(
            patient_id=1,
            prediction=int(prediction),
            status=status,
            probability=float(ckd_probability),
            risk_level=risk_level,
            confidence=float(confidence),
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchPatientData):
    """Predict CKD for multiple patients"""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        predictions = []
        
        for idx, patient in enumerate(batch.patients):
            # Prepare features
            features = prepare_features(patient)
            
            # Random Forest works with raw features - no scaling needed
            # Make prediction directly with raw features
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            ckd_probability = probabilities[1]
            confidence = max(probabilities)
            risk_level = get_risk_level(ckd_probability)
            
            # Get clear status and message
            status, message = get_prediction_message(int(prediction), ckd_probability, risk_level)
            
            predictions.append(PredictionResponse(
                patient_id=idx + 1,
                prediction=int(prediction),
                status=status,
                probability=float(ckd_probability),
                risk_level=risk_level,
                confidence=float(confidence),
                message=message
            ))
        
        # Calculate summary statistics
        total_patients = len(predictions)
        ckd_cases = sum(1 for p in predictions if p.prediction == 1)
        high_risk = sum(1 for p in predictions if p.risk_level == "High")
        avg_probability = sum(p.probability for p in predictions) / total_patients
        
        summary = {
            "total_patients": total_patients,
            "predicted_ckd_cases": ckd_cases,
            "high_risk_patients": high_risk,
            "average_ckd_probability": round(avg_probability, 4),
            "ckd_rate": round((ckd_cases / total_patients) * 100, 2)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/medical-assessment", response_model=LLMAssessmentResponse)
async def get_medical_assessment(request: LLMAssessmentRequest):
    """
    Get detailed medical assessment using LLM (Gemini)
    This endpoint provides expert-level clinical interpretation of the case
    """
    
    print(f"🔍 Debug: Medical assessment endpoint called")
    print(f"🔍 Debug: gemini_client state: {'Available' if gemini_client else 'None'}")
    print(f"🔍 Debug: GEMINI_API_KEY present: {'Yes' if GEMINI_API_KEY else 'No'}")
    
    # Check for client initialization instead of just the key
    if gemini_client is None:
        print("❌ Debug: Returning 503 - gemini_client is None")
        raise HTTPException(
            status_code=503, 
            detail="LLM service not configured or failed to initialize. Please ensure GEMINI_API_KEY environment variable is set correctly."
        )
    
    try:
        # Create medical prompt
        prompt = create_medical_prompt(
            request.patient_data, 
            request.model_prediction, 
            request.additional_context
        )
        
        # >>> REPLACED call_gemini_api with the SDK wrapper <<<
        llm_response = await call_gemini_api_sdk_async(prompt)
        
        # Parse and structure the response
        assessment = parse_llm_response(llm_response)
        
        return assessment
        
    except HTTPException:
        # Re-raise exceptions raised by the LLM wrapper
        raise
    except Exception as e:
        # Catch any other unexpected error
        raise HTTPException(status_code=500, detail=f"Medical assessment error: {str(e)}")

@app.post("/predict-with-assessment")
async def predict_with_medical_assessment(
    patient: PatientData, 
    additional_context: Optional[str] = None
):
    """
    Combined endpoint: Get ML prediction + LLM medical assessment
    This provides both the model prediction and expert clinical interpretation
    """
    
    try:
        # First get the ML prediction
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Prepare features and predict
        features = prepare_features(patient)
        
        # Random Forest works with raw features - no scaling needed
        # Make prediction directly with raw features
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        ckd_probability = probabilities[1]
        confidence = max(probabilities)
        risk_level = get_risk_level(ckd_probability)
        status, message = get_prediction_message(int(prediction), ckd_probability, risk_level)
        
        ml_prediction = PredictionResponse(
            patient_id=1,
            prediction=int(prediction),
            status=status,
            probability=float(ckd_probability),
            risk_level=risk_level,
            confidence=float(confidence),
            message=message
        )
        
        # Get LLM assessment if client is available
        llm_assessment = None
        if gemini_client is not None:
            try:
                assessment_request = LLMAssessmentRequest(
                    patient_data=patient,
                    model_prediction=ml_prediction,
                    additional_context=additional_context
                )
                
                # REPLACED get_medical_assessment with direct logic to use the new SDK
                
                # 1. Create prompt
                prompt = create_medical_prompt(
                    assessment_request.patient_data, 
                    assessment_request.model_prediction, 
                    assessment_request.additional_context
                )
                
                # 2. Call SDK
                llm_response = await call_gemini_api_sdk_async(prompt)
                
                # 3. Parse response
                llm_assessment = parse_llm_response(llm_response)
                
            except HTTPException as e:
                # If LLM fails, still return ML prediction
                llm_assessment = {"error": f"LLM assessment failed: {e.detail}"}
            except Exception as e:
                llm_assessment = {"error": f"Unexpected LLM error: {str(e)}"}
        
        return {
            "ml_prediction": ml_prediction,
            "medical_assessment": llm_assessment,
            "combined_analysis": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "ml_confidence": confidence,
                "assessment_available": llm_assessment is not None and (isinstance(llm_assessment, dict) and "error" not in llm_assessment or isinstance(llm_assessment, LLMAssessmentResponse))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Combined prediction error: {str(e)}")

@app.get("/sample-data")
async def get_sample_data():
    """Get sample patient data for testing"""
    return {
        "sample_patient_healthy": {
            "age": 45.0,
            "weight": 70.0,
            "creatinine": 80.0,
            "sodium": 140.0,
            "potassium": 4.0,
            "glucose": 5.5,
            "urea": 5.0,
            "gender_male": 1
        },
        "sample_patient_ckd": {
            "age": 65.0,
            "weight": 85.0,
            "creatinine": 400.0,
            "sodium": 135.0,
            "potassium": 5.5,
            "glucose": 8.0,
            "urea": 25.0,
            "gender_male": 0
        },
        "note": "Use these sample data points to test the API endpoints. Features match: ['Age of Patient (year)', 'Weight (kg)', 'Creatinine (Umol/L)', 'Sodium (mmol/L)', 'Potassium (mmol/L)', 'Glucose (mmol/L)', 'Urea (mmol/L)', 'Gender_Male']",
        "llm_endpoints": {
            "medical_assessment": "/medical-assessment",
            "predict_with_assessment": "/predict-with-assessment", 
            "note": "Set GEMINI_API_KEY environment variable to use LLM medical assessment features"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting CKD Prediction API Server with LLM Medical Assessment...")
    print("📊 Visit http://localhost:8000/docs for interactive API documentation")
    print("🩺 Visit http://localhost:8000/sample-data for sample test data")
    print("🤖 LLM Assessment:", "✅ Enabled" if GEMINI_API_KEY else "❌ Disabled (set GEMINI_API_KEY)")
    
    # Assuming the file is named ckd_api.py
    uvicorn.run("ckd_api:app", host="0.0.0.0", port=8000, reload=True)