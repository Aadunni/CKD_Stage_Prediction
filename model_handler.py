#!/usr/bin/env python3
"""
Model Handler for CKD Prediction
Handles model loading and predictions without FastAPI
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

class CKDModelHandler:
    """Handler for CKD prediction model"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.gemini_client = None
        self.load_model()
        self.initialize_gemini()
    
    def load_model(self):
        """Load the trained model and feature names"""
        try:
            models_dir = Path('models')
            
            # Look for the latest Random Forest model
            rf_model_files = list(models_dir.glob('best_model_random_forest_*.pkl'))
            feature_files = list(models_dir.glob('feature_names_*.pkl'))
            
            if not rf_model_files:
                # Fallback to old naming convention
                self.model = joblib.load('models/random_forest_model.pkl')
                self.feature_names = joblib.load('models/feature_names.pkl')
                print("📁 Loaded legacy model files")
            else:
                # Use the latest timestamped files
                latest_model = max(rf_model_files, key=lambda x: x.name)
                latest_features = max(feature_files, key=lambda x: x.name)
                
                self.model = joblib.load(latest_model)
                self.feature_names = joblib.load(latest_features)
                print(f"📁 Loaded model: {latest_model.name}")
            
            print("✅ Model loaded successfully!")
            print(f"🏷️  Model type: {type(self.model).__name__}")
            print(f"📋 Features ({len(self.feature_names)}): {self.feature_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def initialize_gemini(self):
        """Initialize Gemini client for medical assessments"""
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            try:
                self.gemini_client = genai.Client()
                print("✅ Gemini Client initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                self.gemini_client = None
        else:
            print("ℹ️  GEMINI_API_KEY not found - medical assessments unavailable")
    
    def prepare_features(self, patient_data: Dict[str, float]) -> np.ndarray:
        """Convert patient data to model features"""
        features = [
            patient_data["age"],
            patient_data["weight"],
            patient_data["creatinine"],
            patient_data["sodium"],
            patient_data["potassium"],
            patient_data["glucose"],
            patient_data["urea"],
            patient_data["gender_male"]
        ]
        return np.array(features).reshape(1, -1)
    
    def calculate_egfr(self, creatinine: float, age: float, is_male: bool) -> float:
        """
        Calculate eGFR using CKD-EPI equation
        
        Args:
            creatinine: Serum creatinine in μmol/L
            age: Age in years
            is_male: True if male, False if female
        
        Returns:
            eGFR in mL/min/1.73m²
        """
        # Convert creatinine from μmol/L to mg/dL
        creatinine_mg_dl = creatinine / 88.4
        
        # CKD-EPI equation
        kappa = 0.9 if is_male else 0.7
        alpha = -0.411 if is_male else -0.329
        gender_factor = 1.0 if is_male else 1.018
        
        min_ratio = min(creatinine_mg_dl / kappa, 1.0)
        max_ratio = max(creatinine_mg_dl / kappa, 1.0)
        
        egfr = 141 * (min_ratio ** alpha) * (max_ratio ** -1.209) * (0.993 ** age) * gender_factor
        
        return round(egfr, 1)
    
    def get_ckd_stage_from_egfr(self, egfr: float) -> str:
        """
        Determine CKD stage based on eGFR
        
        Returns:
            CKD stage description
        """
        if egfr >= 90:
            return "Stage 1 (Normal or high)"
        elif egfr >= 60:
            return "Stage 2 (Mildly decreased)"
        elif egfr >= 45:
            return "Stage 3a (Mild to moderately decreased)"
        elif egfr >= 30:
            return "Stage 3b (Moderately to severely decreased)"
        elif egfr >= 15:
            return "Stage 4 (Severely decreased)"
        else:
            return "Stage 5 (Kidney failure)"
    
    def get_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def predict(self, patient_data: Dict[str, float]) -> Dict[str, Any]:
        """Make a single prediction"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Prepare features
        features = self.prepare_features(patient_data)
        
        # Make prediction
        prediction = int(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        
        # Get probability of CKD (class 1)
        ckd_probability = float(probabilities[1])
        confidence = float(max(probabilities))
        
        # Determine risk level
        risk_level = self.get_risk_level(ckd_probability)
        
        # Calculate eGFR
        is_male = patient_data["gender_male"] == 1
        egfr = self.calculate_egfr(
            patient_data["creatinine"],
            patient_data["age"],
            is_male
        )
        ckd_stage_by_egfr = self.get_ckd_stage_from_egfr(egfr)
        
        # Create message based on risk severity
        if ckd_probability >= 0.7:
            status = "High Severity Risk"
            message = f"⚠️ High severity risk detected ({ckd_probability:.1%} probability). Immediate medical attention is strongly recommended for comprehensive kidney function evaluation."
        elif ckd_probability >= 0.3:
            status = "Moderate Severity Risk"
            message = f"⚠️ Moderate severity risk detected ({ckd_probability:.1%} probability). Medical evaluation recommended to assess kidney function and determine appropriate care."
        else:
            status = "Lower Severity Risk"
            message = f"✅ Lower severity risk indicated ({ckd_probability:.1%} probability). Continue regular monitoring and follow-up care with your healthcare provider."
        
        return {
            "prediction": prediction,
            "probability": ckd_probability,
            "status": status,
            "risk_level": risk_level,
            "confidence": confidence,
            "message": message,
            "egfr": egfr,
            "ckd_stage_by_egfr": ckd_stage_by_egfr
        }
    
    def predict_batch(self, patients: List[Dict[str, float]]) -> Dict[str, Any]:
        """Make batch predictions"""
        predictions = []
        
        for i, patient_data in enumerate(patients):
            result = self.predict(patient_data)
            result["patient_id"] = i + 1
            predictions.append(result)
        
        # Calculate summary statistics
        total_patients = len(predictions)
        ckd_cases = sum(1 for p in predictions if p["prediction"] == 1)
        high_risk = sum(1 for p in predictions if p["risk_level"] == "High")
        ckd_rate = round((ckd_cases / total_patients) * 100, 1) if total_patients > 0 else 0
        
        summary = {
            "total_patients": total_patients,
            "predicted_ckd_cases": ckd_cases,
            "high_risk_patients": high_risk,
            "ckd_rate": ckd_rate
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
    
    def get_medical_assessment(self, patient_data: Dict[str, float], 
                               prediction_result: Dict[str, Any],
                               additional_context: Optional[str] = None) -> Dict[str, Any]:
        """Get AI medical assessment using Gemini"""
        
        if self.gemini_client is None:
            return {
                "error": "Gemini API not available. Please configure GEMINI_API_KEY."
            }
        
        # Retry logic for API calls
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Build prompt
                prompt = self._build_medical_prompt(patient_data, prediction_result, additional_context)
                prompt += f"\n**eGFR Calculation:**\n- eGFR: {prediction_result['egfr']} mL/min/1.73m²\n- CKD Stage by eGFR: {prediction_result['ckd_stage_by_egfr']}\n"
                
                # Call Gemini
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                
                # Parse response
                assessment = self._parse_medical_response(response.text)
                
                # Verify that we got a valid summary
                if assessment.get('summary') and len(assessment.get('summary', '').strip()) > 10:
                    return assessment
                else:
                    # If summary is missing or too short, retry
                    retry_count += 1
                    if retry_count >= max_retries:
                        return {
                            "error": "Failed to generate valid clinical summary after multiple attempts.",
                            "clinical_interpretation": assessment.get('clinical_interpretation', 'No data'),
                            "summary": "Summary generation failed. Please try again."
                        }
                    continue
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    return {
                        "error": f"Failed to get medical assessment after {max_retries} attempts: {str(e)}"
                    }
                # Wait a bit before retrying
                import time
                time.sleep(1)
    
    def _build_medical_prompt(self, patient_data: Dict[str, Any], 
                             prediction_result: Dict[str, Any],
                             additional_context: Optional[str]) -> str:
        """Build prompt for medical assessment"""
        
        gender = "Male" if patient_data["gender_male"] == 1 else "Female"
        
        prompt = f"""You are an experienced nephrologist reviewing a Chronic Kidney Disease (CKD) risk assessment.

**Patient Information:**
- Age: {patient_data['age']} years
- Gender: {gender}
- Weight: {patient_data['weight']} kg

**Laboratory Values:**
- Creatinine: {patient_data['creatinine']} μmol/L (Normal: Male <120, Female <110)
- Sodium: {patient_data['sodium']} mmol/L (Normal: 135-145)
- Potassium: {patient_data['potassium']} mmol/L (Normal: 3.5-5.5)
- Glucose: {patient_data['glucose']} mmol/L (Normal: 3.9-6.1 fasting)
- Urea: {patient_data['urea']} mmol/L (Normal: 2.5-8.0)

**ML Model CKD Risk Assessment:**
- Risk Probability: {prediction_result['probability']:.1%}
- Risk Level: {prediction_result['risk_level']}
- Model Confidence: {prediction_result['confidence']:.1%}

**eGFR Calculation (CKD-EPI Equation):**
- eGFR: {prediction_result['egfr']} mL/min/1.73m²
- CKD Stage by eGFR: {prediction_result['ckd_stage_by_egfr']}

**Assessment Context:**
- ML Model Risk Assessment: {prediction_result['probability']:.1%} probability ({prediction_result['risk_level']} risk)
- eGFR Indicates: {prediction_result['ckd_stage_by_egfr']}
"""

        if additional_context:
            prompt += f"\n**Additional Medical Context:**\n{additional_context}\n"

        prompt += """
Please provide a comprehensive medical assessment in the following structured format:

1. **Clinical Interpretation**: Interpret the lab values in clinical context
2. **Lab Values Analysis**: Detailed analysis of abnormal values and their significance
3. **eGFR vs ML Risk Assessment Comparison**: Compare and analyze the agreement/disagreement between the eGFR-based CKD stage and the ML model's risk probability. Explain any discrepancies and what they might indicate about the patient's kidney health.
4. **Model Validation**: Clinical validation of the ML model's risk assessment considering the eGFR and other clinical indicators
5. **Risk Factors**: List key risk factors identified (as bullet points)
6. **Recommendations**: Treatment and management recommendations based on both the eGFR stage and risk assessment (as bullet points)
7. **Additional Tests**: Suggested additional tests to confirm kidney function status (as bullet points)
8. **Confidence in Model**: Your assessment of the model's risk prediction accuracy given the eGFR calculation
9. **Summary**: Brief overall assessment integrating both the ML risk assessment probability and eGFR results. Focus on the clinical picture rather than definitive staging.

CRITICAL INSTRUCTIONS:
- NEVER use the phrase "Stage 5" or "CKD Stage 5" when discussing the ML model results
- Always refer to the ML model as providing a "risk assessment" or "risk probability" 
- Use terms like "high/moderate/low severity risk" or "X% risk probability" instead of "Stage 5 prediction"
- The ML model does NOT diagnose stages - it provides a severity risk score
- Focus on the eGFR-based staging as the primary clinical staging tool
- When comparing, say "eGFR indicates Stage X while ML model shows Y% risk probability" NOT "eGFR indicates Stage X while ML predicts Stage 5"

Format each section clearly with headers.
"""
        return prompt
    
    def _parse_medical_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the Gemini response into structured format"""
        
        sections = {
            "clinical_interpretation": "",
            "lab_values_analysis": "",
            "egfr_vs_ml_comparison": "",
            "model_validation": "",
            "risk_factors": [],
            "recommendations": [],
            "additional_tests_suggested": [],
            "confidence_in_model": "",
            "summary": ""
        }
        
        current_section = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for section headers (more flexible matching)
            line_lower = line.lower()
            
            # Remove markdown formatting from headers
            clean_line = line.replace('**', '').replace('##', '').replace('#', '').strip()
            clean_lower = clean_line.lower()
            
            if "clinical interpretation" in clean_lower or clean_lower.startswith("1.") and "clinical" in clean_lower:
                current_section = "clinical_interpretation"
                continue
            elif "lab values analysis" in clean_lower or clean_lower.startswith("2.") and "lab" in clean_lower:
                current_section = "lab_values_analysis"
                continue
            elif "egfr vs ml" in clean_lower or "comparison" in clean_lower or clean_lower.startswith("3."):
                if current_section != "clinical_interpretation":  # Avoid false match
                    current_section = "egfr_vs_ml_comparison"
                    continue
            elif "model validation" in clean_lower or clean_lower.startswith("4.") and "validation" in clean_lower:
                current_section = "model_validation"
                continue
            elif "risk factors" in clean_lower or clean_lower.startswith("5.") and "risk" in clean_lower:
                current_section = "risk_factors"
                continue
            elif "recommendations" in clean_lower or clean_lower.startswith("6.") and "recommend" in clean_lower:
                current_section = "recommendations"
                continue
            elif "additional tests" in clean_lower or clean_lower.startswith("7.") and "test" in clean_lower:
                current_section = "additional_tests_suggested"
                continue
            elif "confidence in model" in clean_lower or clean_lower.startswith("8.") and "confidence" in clean_lower:
                current_section = "confidence_in_model"
                continue
            elif "summary" in clean_lower or clean_lower.startswith("9.") and "summary" in clean_lower:
                current_section = "summary"
                continue
            
            # Add content to current section
            if current_section:
                if current_section in ["risk_factors", "recommendations", "additional_tests_suggested"]:
                    if line.startswith(('•', '-', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                        clean_line = line.lstrip('•-*0123456789. ')
                        if clean_line:
                            sections[current_section].append(clean_line)
                else:
                    sections[current_section] += line + " "
        
        # Clean up text sections
        for key in ["clinical_interpretation", "lab_values_analysis", "egfr_vs_ml_comparison", "model_validation", "confidence_in_model", "summary"]:
            sections[key] = sections[key].strip()
            
            # Remove any "Stage 5" references and replace with appropriate language
            if sections[key]:
                # Replace various Stage 5 mentions (case-insensitive and comprehensive)
                import re
                
                # Pattern 1: "prediction of CKD Stage 5" or "probability of CKD Stage 5"
                sections[key] = re.sub(
                    r'(prediction|probability)\s+of\s+(CKD\s+)?[Ss]tage\s+5',
                    r'\1 showing high severity risk',
                    sections[key]
                )
                
                # Pattern 2: "predicts/predicted CKD Stage 5"
                sections[key] = re.sub(
                    r'predict[s|ed]*\s+(a\s+\d+\.?\d*%\s+)?probability\s+of\s+(CKD\s+)?[Ss]tage\s+5',
                    r'indicates \1high severity risk',
                    sections[key]
                )
                
                # Pattern 3: "CKD Stage 5" standalone
                sections[key] = re.sub(
                    r'\b(CKD\s+)?[Ss]tage\s+5\b',
                    r'high severity CKD risk',
                    sections[key]
                )
                
                # Pattern 4: "ML model's prediction" -> "ML model's risk assessment"
                sections[key] = sections[key].replace("ML model's prediction", "ML model's risk assessment")
                sections[key] = sections[key].replace("model's prediction", "model's risk assessment")
                sections[key] = sections[key].replace("model predicts", "model indicates")
                sections[key] = sections[key].replace("model predicted", "model indicated")
                
                # Pattern 5: Clean up any remaining awkward phrasings
                sections[key] = re.sub(
                    r'severe prediction of high severity CKD risk',
                    r'high severity risk assessment',
                    sections[key]
                )
                sections[key] = re.sub(
                    r'prediction of high severity CKD risk',
                    r'high severity risk assessment',
                    sections[key]
                )
        
        # If summary is still empty, try to create one from clinical_interpretation
        if not sections["summary"] and sections["clinical_interpretation"]:
            # Take first 2-3 sentences as summary
            sentences = sections["clinical_interpretation"].split('.')
            summary_sentences = [s.strip() + '.' for s in sentences[:2] if s.strip()]
            sections["summary"] = ' '.join(summary_sentences)
        
        return sections


# Global singleton instance
_model_handler = None

def get_model_handler() -> CKDModelHandler:
    """Get or create the global model handler instance"""
    global _model_handler
    if _model_handler is None:
        _model_handler = CKDModelHandler()
    return _model_handler
