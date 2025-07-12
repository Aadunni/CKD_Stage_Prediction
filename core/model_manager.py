import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List , TYPE_CHECKING
import logging
from pathlib import Path

from models.schemas import PatientData

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model loading, preprocessing, and predictions"""
    
    def __init__(self, model_path: str = "/workspaces/CKD_Stage_Prediction/model_artifacts/RandomForest.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = [
            'Sodium (mmol/L)', 'Potassium (mmol/L)', 'Glucose (mmol/L)',
            'Urea (mmol/L)', 'Body temperature (0^c)', 'SBP', 'DBP', 
            'Weight (kg)', 'Age of Patient', 'Gender_Female', 'Gender_Male', 
            'Creatinine (mg/dl)'
        ]
        self.stage_descriptions = {
            0: "Normal/Minimal Risk",
            1: "Stage 1 - Mild CKD",
            2: "Stage 2 - Moderate CKD", 
            3: "Stage 3 - Moderate to Severe CKD",
            4: "Stage 4 - Severe CKD",
            5: "Stage 5 - End-Stage CKD"
        }
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, patient_data: PatientData) -> np.ndarray:
        """Preprocess patient data for model input"""
        try:
            # Create feature dictionary
            features = {
                'Sodium (mmol/L)': patient_data.sodium,
                'Potassium (mmol/L)': patient_data.potassium,
                'Glucose (mmol/L)': patient_data.glucose,
                'Urea (mmol/L)': patient_data.urea,
                'Body temperature (0^c)': patient_data.body_temperature,
                'SBP': patient_data.sbp,
                'DBP': patient_data.dbp,
                'Weight (kg)': patient_data.weight,
                'Age of Patient': patient_data.age,
                'Gender_Female': 1 if patient_data.gender == "female" else 0,
                'Gender_Male': 1 if patient_data.gender == "male" else 0,
                'Creatinine (mg/dl)': patient_data.creatinine
            }
            
            # Convert to DataFrame with correct column order
            df = pd.DataFrame([features])
            df = df[self.feature_columns]  # Ensure correct order
            
            return df.values
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def predict(self, patient_data: PatientData) -> Tuple[int, np.ndarray]:
        """Make prediction"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess input
            X = self.preprocess_input(patient_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            return int(prediction), probabilities
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_risk_level(self, stage: int) -> str:
        """Determine risk level based on CKD stage"""
        risk_mapping = {
            0: "Low",
            1: "Low",
            2: "Moderate",
            3: "High",
            4: "Very High",
            5: "Critical"
        }
        return risk_mapping.get(stage, "Unknown")
    
    def get_recommendations(self, stage: int) -> List[str]:
        """Get clinical recommendations based on CKD stage"""
        recommendations = {
            0: [
                "Maintain healthy lifestyle",
                "Regular health checkups",
                "Monitor blood pressure and glucose"
            ],
            1: [
                "Lifestyle modifications recommended",
                "Regular monitoring of kidney function",
                "Control blood pressure and diabetes if present"
            ],
            2: [
                "Consult nephrologist",
                "Strict blood pressure control",
                "Dietary protein restriction may be needed"
            ],
            3: [
                "Urgent nephrology consultation required",
                "Prepare for renal replacement therapy",
                "Management of mineral and bone disorders"
            ],
            4: [
                "Immediate nephrology care",
                "Dialysis preparation",
                "Transplant evaluation"
            ],
            5: [
                "Critical - Immediate medical attention",
                "Dialysis or transplant required",
                "Intensive medical management"
            ]
        }
        return recommendations.get(stage, ["Consult healthcare provider"])