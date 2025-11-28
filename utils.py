#!/usr/bin/env python3
"""
Streamlit Utility Functions
Contains all business logic and helper functions for the CKD Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from model_handler import get_model_handler

# Normal ranges for lab values (for reference and validation)
NORMAL_RANGES = {
    "age": {"min": 0, "max": 120, "unit": "years"},
    "weight": {"min": 20, "max": 200, "unit": "kg"},
    "creatinine": {"male": {"max": 120}, "female": {"max": 110}, "unit": "μmol/L"},
    "sodium": {"min": 135, "max": 145, "unit": "mmol/L"},
    "potassium": {"min": 3.5, "max": 5.5, "unit": "mmol/L"},
    "glucose": {"min": 3.9, "max": 6.1, "unit": "mmol/L (fasting)"},
    "urea": {"min": 2.5, "max": 8.0, "unit": "mmol/L"}
}


@st.cache_resource
def load_model():
    """Load and cache the model handler"""
    return get_model_handler()


def check_model_loaded() -> bool:
    """Check if model is loaded"""
    try:
        handler = load_model()
        return handler.model is not None
    except:
        return False


def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        "Low": "#00C851",
        "Medium": "#ffbb33", 
        "High": "#ff4444"
    }
    return colors.get(risk_level, "#6c757d")


def create_gauge_chart(probability, risk_level):
    """Create a gauge chart for CKD probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "CKD Risk Probability (%)"},
        delta = {'reference': 30},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_risk_color(risk_level)},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_lab_values_chart(patient_data):
    """Create a radar chart for lab values comparison with normal ranges"""
    
    # Normalize values to 0-100 scale for comparison
    categories = ['Creatinine', 'Sodium', 'Potassium', 'Glucose', 'Urea']
    
    # Calculate normalized values (higher = more abnormal)
    gender = "male" if patient_data["gender_male"] == 1 else "female"
    
    values = []
    normal_values = []
    
    # Creatinine
    creat_max = NORMAL_RANGES["creatinine"][gender]["max"]
    creat_norm = min(100, (patient_data["creatinine"] / creat_max) * 100)
    values.append(creat_norm)
    normal_values.append(100)  # Normal threshold
    
    # Sodium (distance from normal range)
    sodium_center = (NORMAL_RANGES["sodium"]["min"] + NORMAL_RANGES["sodium"]["max"]) / 2
    sodium_range = NORMAL_RANGES["sodium"]["max"] - NORMAL_RANGES["sodium"]["min"]
    sodium_norm = abs(patient_data["sodium"] - sodium_center) / sodium_range * 100
    values.append(min(100, sodium_norm))
    normal_values.append(20)  # 20% deviation as threshold
    
    # Potassium
    potass_center = (NORMAL_RANGES["potassium"]["min"] + NORMAL_RANGES["potassium"]["max"]) / 2
    potass_range = NORMAL_RANGES["potassium"]["max"] - NORMAL_RANGES["potassium"]["min"]
    potass_norm = abs(patient_data["potassium"] - potass_center) / potass_range * 100
    values.append(min(100, potass_norm))
    normal_values.append(20)
    
    # Glucose
    glucose_max = NORMAL_RANGES["glucose"]["max"]
    glucose_norm = max(0, (patient_data["glucose"] - glucose_max) / glucose_max * 100)
    values.append(min(100, glucose_norm))
    normal_values.append(20)
    
    # Urea
    urea_max = NORMAL_RANGES["urea"]["max"]
    urea_norm = max(0, (patient_data["urea"] - urea_max) / urea_max * 100)
    values.append(min(100, urea_norm))
    normal_values.append(20)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Values',
        line_color='red'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        name='Normal Threshold',
        line_color='green',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Lab Values vs Normal Range",
        height=400
    )
    
    return fig


def display_prediction_result(prediction_data):
    """Display prediction results with styling"""

    # Highlight eGFR prominently with risk level
    egfr_value = prediction_data['egfr']
    risk_level = prediction_data['risk_level']
    
    # Determine eGFR card color based on stage
    if egfr_value < 15:
        egfr_card_style = "background-color: #f8d7da; border: 2px solid #f5c6cb;"
        egfr_status = "⚠️ Kidney Failure"
    elif egfr_value < 30:
        egfr_card_style = "background-color: #fff3cd; border: 2px solid #ffeaa7;"
        egfr_status = "⚠️ Severely Decreased"
    elif egfr_value < 60:
        egfr_card_style = "background-color: #d1ecf1; border: 2px solid #bee5eb;"
        egfr_status = "ℹ️ Decreased Function"
    else:
        egfr_card_style = "background-color: #d4edda; border: 2px solid #c3e6cb;"
        egfr_status = "✅ Normal/Mild"
    
    st.markdown(f"""
    <div style="color: black; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0; {egfr_card_style}">
        <h3>eGFR: {egfr_value} mL/min/1.73m²</h3>
        <p><strong>CKD Stage by eGFR:</strong> {prediction_data['ckd_stage_by_egfr']}</p>
        <p><strong>Risk Level:</strong> <span style="color:{get_risk_color(risk_level)}; font-weight: bold;">{risk_level}</span></p>
        <p><strong>Status:</strong> {egfr_status}</p>
    </div>
    """, unsafe_allow_html=True)

    # Model prediction - focused on diagnosis risk
    probability = prediction_data['probability']
    
    # Determine message based on probability
    if probability >= 0.7:
        probability_msg = f"High risk assessment probability ({probability:.1%}). Immediate medical attention recommended for comprehensive kidney evaluation."
        box_color = "#fff3cd"
        icon = "⚠️"
    elif probability >= 0.3:
        probability_msg = f"Moderate risk assessment probability ({probability:.1%}). Medical evaluation recommended to assess kidney function."
        box_color = "#d1ecf1"
        icon = "ℹ️"
    else:
        probability_msg = f"Lower risk assessment probability ({probability:.1%}). Continue regular monitoring and follow-up care."
        box_color = "#d4edda"
        icon = "✅"
    
    st.markdown(f"""
    <div style="background-color: {box_color}; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; color: black;">
        <h4>{icon} CKD Risk Assessment</h4>
        <p>{probability_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for gauge chart
    col1, col2 = st.columns([1, 1])

    with col1:
        st.metric("Risk Probability", f"{probability:.1%}")

    with col2:
        # Display gauge chart
        gauge_fig = create_gauge_chart(probability, risk_level)
        st.plotly_chart(gauge_fig, use_container_width=True)


def make_single_prediction(patient_data: Dict[str, float]) -> Tuple[Optional[Dict], Optional[str]]:
    """Make a single prediction using the model handler"""
    try:
        handler = load_model()
        result = handler.predict(patient_data)
        return result, None
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"


def make_batch_prediction(patients_data: List[Dict[str, float]]) -> Tuple[Optional[Dict], Optional[str]]:
    """Make a batch prediction using the model handler"""
    try:
        handler = load_model()
        results = handler.predict_batch(patients_data)
        return results, None
    except Exception as e:
        return None, f"Error making batch prediction: {str(e)}"


def get_medical_assessment(patient_data: Dict[str, float], additional_context: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Get medical assessment with prediction using the model handler"""
    try:
        handler = load_model()
        
        # First get prediction
        prediction = handler.predict(patient_data)
        
        # Then get medical assessment
        assessment = handler.get_medical_assessment(patient_data, prediction, additional_context)
        
        result = {
            "ml_prediction": prediction,
            "medical_assessment": assessment
        }
        
        return result, None
    except Exception as e:
        return None, f"Error getting assessment: {str(e)}"


def get_sample_data() -> Tuple[Optional[Dict], Optional[str]]:
    """Get sample data for testing"""
    sample_data = {
        "sample_patient_healthy": {
            "age": 45.0,
            "weight": 70.0,
            "creatinine": 85.0,
            "sodium": 140.0,
            "potassium": 4.0,
            "glucose": 5.5,
            "urea": 5.0,
            "gender_male": 1
        },
        "sample_patient_ckd": {
            "age": 65.0,
            "weight": 75.0,
            "creatinine": 400.0,
            "sodium": 135.0,
            "potassium": 5.8,
            "glucose": 8.0,
            "urea": 25.0,
            "gender_male": 0
        },
        "note": "This model predicts CKD Stage 5 (end-stage renal disease) specifically. A negative prediction does not rule out earlier CKD stages. Required features: age, weight, creatinine, sodium, potassium, glucose, urea, gender_male (1 for male, 0 for female)"
    }
    return sample_data, None


def get_model_info() -> bool:
    """Check if model is loaded"""
    return check_model_loaded()


def create_csv_template():
    """Create a CSV template for batch prediction"""
    template_data = {
        "age": [45, 65, 55],
        "weight": [70, 85, 75],
        "creatinine": [80, 400, 150],
        "sodium": [140, 135, 138],
        "potassium": [4.0, 5.5, 4.5],
        "glucose": [5.5, 8.0, 6.0],
        "urea": [5.0, 25.0, 12.0],
        "gender_male": [1, 0, 1]
    }
    return pd.DataFrame(template_data)


def create_lab_values_table(age, weight, creatinine, sodium, potassium, glucose, urea, gender):
    """Create a dataframe for lab values display"""
    lab_data = {
        "Parameter": ["Age", "Weight", "Creatinine", "Sodium", "Potassium", "Glucose", "Urea"],
        "Value": [
            f"{age} years", 
            f"{weight} kg", 
            f"{creatinine} μmol/L", 
            f"{sodium} mmol/L", 
            f"{potassium} mmol/L", 
            f"{glucose} mmol/L", 
            f"{urea} mmol/L"
        ],
        "Normal Range": [
            "0-120 years", 
            "20-200 kg", 
            f"<{120 if gender=='Male' else 110} μmol/L",
            "135-145 mmol/L", 
            "3.5-5.5 mmol/L", 
            "3.9-6.1 mmol/L", 
            "2.5-8.0 mmol/L"
        ]
    }
    return pd.DataFrame(lab_data)


def create_batch_results_dataframe(predictions):
    """Create a dataframe from batch prediction results"""
    return pd.DataFrame([
        {
            "Patient ID": pred["patient_id"],
            "Prediction": "CKD Stage 5" if pred["prediction"] == 1 else "Not Stage 5",
            "Stage 5 Probability": f"{pred['probability']:.1%}",
            "eGFR": f"{pred['egfr']} mL/min/1.73m²",
            "eGFR Stage": pred["ckd_stage_by_egfr"],
            "Risk Level": pred["risk_level"],
            "Confidence": f"{pred['confidence']:.1%}"
        }
        for pred in predictions
    ])


def create_risk_distribution_chart(predictions_df):
    """Create pie chart for risk level distribution"""
    risk_counts = predictions_df["Risk Level"].value_counts()
    fig = px.pie(
        values=risk_counts.values, 
        names=risk_counts.index,
        title="Risk Level Distribution",
        color_discrete_map={
            "Low": "#00C851",
            "Medium": "#ffbb33",
            "High": "#ff4444"
        }
    )
    return fig


def create_probability_distribution_chart(predictions):
    """Create histogram for probability distribution"""
    probs = [pred["probability"] for pred in predictions]
    fig = px.histogram(x=probs, nbins=20, title="CKD Probability Distribution")
    fig.update_xaxis(title="CKD Probability")
    fig.update_yaxis(title="Number of Patients")
    return fig


def display_medical_assessment(assessment):
    """Display the medical assessment results"""
    if not assessment or "error" in assessment:
        st.warning("LLM Medical Assessment not available. Ensure GEMINI_API_KEY is configured.")
        if assessment and "error" in assessment:
            st.error(assessment["error"])
        return
    
    # Clinical Interpretation
    st.markdown("#### 🩺 Clinical Interpretation")
    st.write(assessment["clinical_interpretation"])
    
    # Lab Values Analysis
    st.markdown("#### 🔬 Lab Values Analysis")
    st.write(assessment["lab_values_analysis"])
    
    # eGFR vs ML Model Comparison
    if assessment.get("egfr_vs_ml_comparison"):
        st.markdown("#### ⚖️ eGFR vs ML Model Comparison")
        st.info(assessment["egfr_vs_ml_comparison"])
    
    # Model Validation
    st.markdown("#### 🤖 Model Validation")
    st.write(assessment["model_validation"])
    
    # Risk Factors and Recommendations in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Risk Factors")
        for factor in assessment["risk_factors"]:
            st.write(f"• {factor}")
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        for rec in assessment["recommendations"]:
            st.write(f"• {rec}")
    
    # Additional Tests
    st.markdown("#### 🧪 Additional Tests Suggested")
    for test in assessment["additional_tests_suggested"]:
        st.write(f"• {test}")
    
    # Confidence and Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Confidence in Model")
        st.info(assessment["confidence_in_model"])
    
    with col2:
        st.markdown("#### 📝 Summary")
        st.success(assessment["summary"])


def get_system_status() -> Dict[str, bool]:
    """Get system status for all components"""
    status = {
        "model_loaded": check_model_loaded(),
        "llm_available": False
    }
    
    # Check LLM availability
    try:
        handler = load_model()
        status["llm_available"] = handler.gemini_client is not None
    except:
        pass
    
    return status
