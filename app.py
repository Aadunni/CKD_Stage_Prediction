"""
CKD Prediction Streamlit App
A comprehensive web interface for Chronic Kidney Disease Stage 5 prediction
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import utility functions
from streamlit_utils import (
    check_model_loaded,
    get_risk_color,
    create_gauge_chart,
    create_lab_values_chart,
    display_prediction_result,
    make_single_prediction,
    make_batch_prediction,
    get_medical_assessment,
    get_sample_data,
    get_model_info,
    create_csv_template,
    create_lab_values_table,
    create_batch_results_dataframe,
    create_risk_distribution_chart,
    create_probability_distribution_chart,
    display_medical_assessment,
    get_system_status,
    NORMAL_RANGES
)

# Page configuration
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .warning-box {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box, .warning-box * {
        color: #000000 !important;
    }
    .success-box {
        background-color: #d4edda !important;
        border: 1px solid #74b9ff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box, .success-box * {
        color: #000000 !important;
    }
    .error-box {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box, .error-box * {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🩺 CKD Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6c757d;">AI-Powered Chronic Kidney Disease Risk Assessment</p>', unsafe_allow_html=True)
    
    # Check model status
    if not check_model_loaded():
        st.error("⚠️ Model could not be loaded. Please ensure the model files exist in the models/ directory")
        st.info("Required files: best_model_random_forest_*.pkl and feature_names_*.pkl")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", [
        "Single Patient Prediction",
        "Batch Prediction", 
        "AI Medical Assessment",
        "Sample Data",
        "About"
    ])
    
    if page == "Single Patient Prediction":
        single_patient_prediction()
    elif page == "Batch Prediction":
        batch_prediction()
    elif page == "AI Medical Assessment":
        medical_assessment_page()
    elif page == "Sample Data":
        sample_data_page()
    elif page == "About":
        about_page()

def single_patient_prediction():
    """Single patient prediction interface"""
    
    st.markdown('<h2 class="sub-header">Single Patient Prediction</h2>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("patient_form"):
        st.subheader("Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            creatinine = st.number_input("Creatinine (μmol/L)", min_value=0.0, value=90.0, step=1.0, 
                                       help="Normal: Male <120, Female <110")
        
        with col2:
            sodium = st.number_input("Sodium (mmol/L)", min_value=120.0, max_value=160.0, value=140.0, step=0.1,
                                   help="Normal: 135-145")
            potassium = st.number_input("Potassium (mmol/L)", min_value=2.0, max_value=8.0, value=4.0, step=0.1,
                                      help="Normal: 3.5-5.5")
            glucose = st.number_input("Glucose (mmol/L)", min_value=2.0, max_value=30.0, value=5.5, step=0.1,
                                    help="Normal: 3.9-6.1 (fasting)")
            urea = st.number_input("Urea (mmol/L)", min_value=1.0, max_value=50.0, value=5.0, step=0.1,
                                 help="Normal: 2.5-8.0")
        
        submitted = st.form_submit_button("🔍 Predict CKD Risk", type="primary")
        
        if submitted:
            # Prepare patient data
            patient_data = {
                "age": age,
                "weight": weight,
                "creatinine": creatinine,
                "sodium": sodium,
                "potassium": potassium,
                "glucose": glucose,
                "urea": urea,
                "gender_male": 1 if gender == "Male" else 0
            }
            
            # Make prediction with medical assessment
            with st.spinner("Making prediction and getting AI assessment..."):
                results, error = get_medical_assessment(patient_data, additional_context=None)
                
                if results:
                    prediction_data = results["ml_prediction"]
                    medical_assessment = results["medical_assessment"]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Lab Values Analysis")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # Display lab values table
                        lab_df = create_lab_values_table(age, weight, creatinine, sodium, potassium, glucose, urea, gender)
                        st.table(lab_df)

                    with col2:
                        # Radar chart
                        radar_fig = create_lab_values_chart(patient_data)
                        st.plotly_chart(radar_fig, use_container_width=True)

                    # Display Gemini summary first (where model card used to be)
                    st.markdown("---")
                    st.subheader("AI Clinical Summary")
                    
                    # Check if medical assessment is available and valid
                    if medical_assessment and isinstance(medical_assessment, dict) and "error" not in medical_assessment:
                        # Display only the summary
                        summary_text = medical_assessment.get('summary', '')
                        if summary_text:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📋 Clinical Summary</h4>
                                <p>{summary_text}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("⚠️ Clinical summary generation failed. Please try again or check your GEMINI_API_KEY configuration.")
                            if medical_assessment.get('clinical_interpretation'):
                                st.info(f"Clinical Interpretation: {medical_assessment.get('clinical_interpretation', '')[:500]}...")
                    else:
                        # Show error details
                        error_msg = medical_assessment.get('error', 'Unknown error') if isinstance(medical_assessment, dict) else 'AI Medical Assessment not available'
                        st.error(f"⚠️ Clinical Summary Generation Error: {error_msg}")
                        st.warning("Please ensure GEMINI_API_KEY is properly configured in your .env file.")
                    
                    # Move prediction results below Gemini summary
                    st.markdown("---")
                    st.subheader("CKD Diagnosis Risk Assessment")
                    display_prediction_result(prediction_data)
                else:
                    st.error(f"Prediction Error: {error}")

def batch_prediction():
    """Batch prediction interface"""
    
    st.markdown('<h2 class="sub-header">Batch Prediction</h2>', unsafe_allow_html=True)
    
    st.info("Upload a CSV file with patient data or use the sample template")
    
    # Option to download template
    if st.button("📥 Download CSV Template"):
        template_df = create_csv_template()
        
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template CSV",
            data=csv,
            file_name="ckd_prediction_template.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Run Batch Prediction", type="primary"):
                # Prepare batch data
                patients = df.to_dict('records')
                
                with st.spinner("Running batch prediction..."):
                    results, error = make_batch_prediction(patients)
                    
                    if results:
                        # Display summary
                        st.subheader("Batch Prediction Results")
                        
                        summary = results["summary"]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Patients", summary["total_patients"])
                        with col2:
                            st.metric("CKD Cases", summary["predicted_ckd_cases"])
                        with col3:
                            st.metric("High Risk", summary["high_risk_patients"])
                        with col4:
                            st.metric("CKD Rate", f"{summary['ckd_rate']}%")
                        
                        # Results table
                        predictions_df = create_batch_results_dataframe(results["predictions"])
                        
                        st.subheader("Detailed Results")
                        st.dataframe(predictions_df)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk distribution pie chart
                            fig_pie = create_risk_distribution_chart(predictions_df)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Probability distribution
                            fig_hist = create_probability_distribution_chart(results["predictions"])
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Download results
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"ckd_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(error)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def medical_assessment_page():
    """Medical assessment with LLM - Full detailed assessment"""
    
    st.markdown('<h2 class="sub-header">AI Medical Assessment</h2>', unsafe_allow_html=True)
    st.info("Get comprehensive detailed medical interpretation using AI (requires GEMINI_API_KEY)")
    
    st.subheader("Patient Information")
    
    with st.form("assessment_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=65.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=75.0, step=0.1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            creatinine = st.number_input("Creatinine (μmol/L)", min_value=0.0, value=250.0, step=1.0)
        
        with col2:
            sodium = st.number_input("Sodium (mmol/L)", min_value=120.0, max_value=160.0, value=135.0, step=0.1)
            potassium = st.number_input("Potassium (mmol/L)", min_value=2.0, max_value=8.0, value=5.2, step=0.1)
            glucose = st.number_input("Glucose (mmol/L)", min_value=2.0, max_value=30.0, value=7.5, step=0.1)
            urea = st.number_input("Urea (mmol/L)", min_value=1.0, max_value=50.0, value=18.0, step=0.1)
        
        additional_context = st.text_area(
            "Additional Medical History (optional)", 
            placeholder="Enter any additional medical history, symptoms, or context...",
            height=100
        )
        
        submitted = st.form_submit_button("🩺 Get Full Medical Assessment", type="primary")
        
        if submitted:
            # Prepare patient data
            patient_data = {
                "age": age,
                "weight": weight,
                "creatinine": creatinine,
                "sodium": sodium,
                "potassium": potassium,
                "glucose": glucose,
                "urea": urea,
                "gender_male": 1 if gender == "Male" else 0
            }
            
            with st.spinner("Getting comprehensive AI medical assessment..."):
                results, error = get_medical_assessment(patient_data, additional_context if additional_context else None)
                
                if results:
                    prediction_data = results["ml_prediction"]
                    medical_assessment = results["medical_assessment"]
                    
                    # Display eGFR and Risk Assessment first
                    st.markdown("---")
                    st.subheader("eGFR & Risk Assessment")
                    display_prediction_result(prediction_data)
                    
                    # Display FULL detailed medical assessment
                    st.markdown("---")
                    st.subheader("Comprehensive AI Medical Assessment")
                    
                    if medical_assessment and isinstance(medical_assessment, dict) and "error" not in medical_assessment:
                        display_medical_assessment(medical_assessment)
                    else:
                        error_msg = medical_assessment.get('error', 'Unknown error') if isinstance(medical_assessment, dict) else 'AI Medical Assessment not available'
                        st.error(f"⚠️ Medical Assessment Error: {error_msg}")
                        st.warning("Please ensure GEMINI_API_KEY is properly configured in your .env file.")
                else:
                    st.error(f"Assessment Error: {error}")

def sample_data_page():
    """Sample data and API information"""
    
    st.markdown('<h2 class="sub-header">Sample Data & API Information</h2>', unsafe_allow_html=True)
    
    data, error = get_sample_data()
    
    if data:
        st.subheader("Sample Patient Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Sample Healthy Patient")
            healthy_data = data["sample_patient_healthy"]
            st.json(healthy_data)
            
            if st.button("Test Healthy Patient", key="test_healthy"):
                with st.spinner("Testing..."):
                    result, err = make_single_prediction(healthy_data)
                    if result:
                        st.success(f"Prediction: {result['status']} (Probability: {result['probability']:.1%})")
                    else:
                        st.error("Test failed")
        
        with col2:
            st.markdown("#### 🚨 Sample CKD Patient")
            ckd_data = data["sample_patient_ckd"]
            st.json(ckd_data)
            
            if st.button("Test CKD Patient", key="test_ckd"):
                with st.spinner("Testing..."):
                    result, err = make_single_prediction(ckd_data)
                    if result:
                        st.warning(f"Prediction: {result['status']} (Probability: {result['probability']:.1%})")
                    else:
                        st.error("Test failed")
        
        # API Endpoints
        st.subheader("Model Information")
        st.write(data["note"])
        
        # Display feature information
        st.subheader("Required Features")
        features_info = pd.DataFrame({
            "Feature": ["age", "weight", "creatinine", "sodium", "potassium", "glucose", "urea", "gender_male"],
            "Description": [
                "Age of patient (years)",
                "Weight (kg)",
                "Creatinine (μmol/L)",
                "Sodium (mmol/L)",
                "Potassium (mmol/L)",
                "Glucose (mmol/L)",
                "Urea (mmol/L)",
                "Gender (1=Male, 0=Female)"
            ]
        })
        st.table(features_info)
    else:
        st.error(error)

def about_page():
    """About page with system information"""
    
    st.markdown('<h2 class="sub-header">About CKD Prediction System</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Purpose
    This system uses machine learning to predict the risk of **Chronic Kidney Disease Stage 5 (End-Stage Renal Disease)** 
    based on laboratory values and patient demographics.
    
    ### 🤖 Model Information
    - **Algorithm**: Random Forest Classifier
    - **Target**: CKD Stage 5 (ESRD) Detection
    - **Features**: 8 clinical parameters (age, weight, lab values)
    - **Output**: Binary classification (CKD Stage 5 vs Not Stage 5)
    - **Additional**: Risk probability and confidence scores
    
    ### ⚠️ Important: What This Model Does NOT Do
    - **Does NOT diagnose all CKD stages**: This model is trained specifically for Stage 5 detection
    - **Does NOT rule out Stages 1-4**: A negative result means "Not Stage 5", not "No kidney disease"
    - **Does NOT replace clinical diagnosis**: Always consult healthcare professionals for comprehensive kidney assessment
    
    ### 📊 Input Parameters
    1. **Age**: Patient age in years (0-120)
    2. **Weight**: Patient weight in kg (20-200)
    3. **Gender**: Male or Female
    4. **Creatinine**: Serum creatinine in μmol/L (Normal: Male <120, Female <110)
    5. **Sodium**: Serum sodium in mmol/L (Normal: 135-145)
    6. **Potassium**: Serum potassium in mmol/L (Normal: 3.5-5.5)
    7. **Glucose**: Blood glucose in mmol/L (Normal: 3.9-6.1 fasting)
    8. **Urea**: Blood urea in mmol/L (Normal: 2.5-8.0)
    
    ### 🩺 Risk Levels (for Stage 5)
    - **Low Risk**: <30% probability of Stage 5
    - **Medium Risk**: 30-70% probability of Stage 5
    - **High Risk**: >70% probability of Stage 5
    
    ### 🔬 eGFR (Estimated Glomerular Filtration Rate)
    The system calculates eGFR using the CKD-EPI equation based on:
    - Serum creatinine (μmol/L)
    - Age (years)
    - Gender
    
    **CKD Stages by eGFR (mL/min/1.73m²):**
    - **Stage 1**: ≥90 (Normal or high)
    - **Stage 2**: 60-89 (Mildly decreased)
    - **Stage 3a**: 45-59 (Mild to moderately decreased)
    - **Stage 3b**: 30-44 (Moderately to severely decreased)
    - **Stage 4**: 15-29 (Severely decreased)
    - **Stage 5**: <15 (Kidney failure / ESRD)
    
    ### 📈 Understanding the Probability
    The probability shown represents the likelihood of **CKD Stage 5** specifically:
    - **High probability (>70%)**: Strong indication of Stage 5 CKD
    - **Medium probability (30-70%)**: Moderate risk, requires medical evaluation
    - **Low probability (<30%)**: Low Stage 5 risk, but earlier stages (1-4) cannot be ruled out
    
    ### �🔬 AI Medical Assessment
    When enabled with GEMINI_API_KEY, the system provides:
    - Clinical interpretation of lab values
    - Risk factor analysis
    - Treatment recommendations
    - Suggested additional tests
    - Model validation from clinical perspective
    
    ### ⚠️ Important Disclaimer
    This system is for educational and research purposes only. It is designed to detect CKD Stage 5 
    (end-stage renal disease) and does not diagnose or rule out earlier stages of kidney disease. 
    Always consult with qualified healthcare professionals for medical diagnosis, comprehensive kidney 
    function assessment, and treatment decisions.
    """)
    
    # System status
    st.subheader("System Status")
    
    status = get_system_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if status["model_loaded"]:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
    
    with col2:
        if status["llm_available"]:
            st.success("✅ AI Assessment Available")
        else:
            st.warning("⚠️ AI Assessment Unavailable")

if __name__ == "__main__":
    main()