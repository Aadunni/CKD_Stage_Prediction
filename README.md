# 🩺 Chronic Kidney Disease (CKD) Stage 5 Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered web application for predicting Chronic Kidney Disease Stage 5 using machine learning, with integrated LLM-based medical assessment capabilities.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Model Information](#-model-information)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🎯 Overview

This system provides an end-to-end solution for predicting Chronic Kidney Disease (CKD) Stage 5 based on laboratory values and patient demographics. It combines:

- **Machine Learning Predictions**: Random Forest classifier trained on real hospital data
- **AI Medical Assessment**: Gemini-powered clinical interpretation and recommendations
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **REST API**: FastAPI backend for programmatic access

### Key Capabilities

✅ **Single & Batch Predictions** - Predict CKD risk for one or multiple patients  
✅ **Risk Stratification** - Automatic categorization into Low, Medium, High risk levels  
✅ **Clinical Interpretation** - AI-powered medical assessment using Google Gemini  
✅ **Visual Analytics** - Interactive charts and probability gauges  
✅ **RESTful API** - Easy integration with other healthcare systems  

---

## ✨ Features

### 🔬 **Medical Features**
- Predicts CKD Stage 5 based on 8 clinical parameters
- Gender-specific analysis (Male/Female)
- Real-time probability calculation
- Risk level assessment (Low: <30%, Medium: 30-70%, High: >70%)

### 🤖 **AI-Powered Assessment**
- Clinical interpretation of lab values
- Detailed risk factor analysis
- Treatment recommendations
- Suggested additional tests
- Model validation from medical perspective

### 📊 **Visualization & Analytics**
- Probability gauge charts
- Lab values comparison radar charts
- Batch prediction summaries
- Risk distribution pie charts
- Downloadable CSV reports

### 🔌 **API Features**
- RESTful endpoints for predictions
- Batch processing support
- Medical assessment integration
- Comprehensive API documentation (Swagger/OpenAPI)
- Sample data endpoints for testing

---

## 🏗️ System Architecture

```
┌─────────────────┐
│   Streamlit UI  │ (Port 8501)
│  (Frontend)     │
└────────┬────────┘
         │
         │ HTTP Requests
         ↓
┌─────────────────┐
│   FastAPI       │ (Port 8000)
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ↓         ↓
┌─────────┐ ┌──────────────┐
│ Random  │ │ Google Gemini│
│ Forest  │ │ LLM (AI)     │
│ Model   │ │ Assessment   │
└─────────┘ └──────────────┘
```

---

## 🛠️ Technology Stack

### **Backend**
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.10+** - Core programming language
- **scikit-learn** - Machine learning model (Random Forest)
- **pandas & numpy** - Data processing

### **Frontend**
- **Streamlit** - Interactive web dashboard
- **Plotly** - Data visualization
- **Seaborn & Matplotlib** - Statistical graphics

### **AI/ML Libraries**
- **Google Generative AI (Gemini)** - Medical assessment and interpretation
- **XGBoost & CatBoost** - Additional ML models (for experimentation)

### **Data Management**
- **joblib** - Model serialization
- **python-dotenv** - Environment variable management

---

## 📦 Installation

### **Prerequisites**

- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Ayo-Cyber/Chronic-Kidney-Disease.git
cd Chronic-Kidney-Disease
```

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### **Step 3: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### **Environment Variables**

Create a `.env` file in the root directory:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
API_URL=http://localhost:8000
```

### **Get Gemini API Key**

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

> **Note:** The system works without the Gemini API key, but AI medical assessment features will be disabled.

---

## 🚀 Running the Application

### **Option 1: Quick Start (Recommended)**

Use the provided startup script:

```bash
# Start API server
sh start_api_with_llm.sh
```

In a new terminal, start the Streamlit app:

```bash
# Activate virtual environment first
source venv/bin/activate

# Start Streamlit
streamlit run app.py
```

### **Option 2: Manual Start**

**Terminal 1 - API Server:**

```bash
source venv/bin/activate
python ckd_api.py
```

**Terminal 2 - Streamlit App:**

```bash
source venv/bin/activate
streamlit run app.py
```

### **Access the Application**

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Root**: http://localhost:8000

---

## 📚 API Documentation

### **Base URL**
```
http://localhost:8000
```

### **Endpoints**

#### **1. Health Check**
```http
GET /
```
Returns API status and model information.

#### **2. Single Prediction**
```http
POST /predict
```

**Request Body:**
```json
{
  "age": 65.0,
  "weight": 85.0,
  "creatinine": 400.0,
  "sodium": 135.0,
  "potassium": 5.5,
  "glucose": 8.0,
  "urea": 25.0,
  "gender_male": 0
}
```

**Response:**
```json
{
  "patient_id": 1,
  "prediction": 1,
  "status": "CKD Detected",
  "probability": 0.95,
  "risk_level": "High",
  "confidence": 0.95,
  "message": "⚠️ Chronic Kidney Disease Stage 5 detected..."
}
```

#### **3. Batch Prediction**
```http
POST /predict-batch
```

**Request Body:**
```json
{
  "patients": [
    {
      "age": 45.0,
      "weight": 70.0,
      "creatinine": 80.0,
      "sodium": 140.0,
      "potassium": 4.0,
      "glucose": 5.5,
      "urea": 5.0,
      "gender_male": 1
    },
    {
      "age": 65.0,
      "weight": 85.0,
      "creatinine": 400.0,
      "sodium": 135.0,
      "potassium": 5.5,
      "glucose": 8.0,
      "urea": 25.0,
      "gender_male": 0
    }
  ]
}
```

#### **4. Medical Assessment**
```http
POST /predict-with-assessment
```
Returns ML prediction + AI-powered medical interpretation (requires Gemini API key).

#### **5. Sample Data**
```http
GET /sample-data
```
Returns sample patient data for testing.

### **Interactive API Documentation**

Visit http://localhost:8000/docs for full Swagger UI documentation with:
- Try-it-out functionality
- Request/response schemas
- Authentication details
- Example requests

---

## 🧠 Model Information

### **Algorithm**
- **Type**: Random Forest Classifier
- **Trees**: 100 estimators
- **Max Depth**: 10
- **Training Accuracy**: 99.75%
- **Test Accuracy**: 96.12%
- **AUC-ROC**: 99.65%

### **Input Features (8 parameters)**

| Feature | Description | Unit | Normal Range |
|---------|-------------|------|--------------|
| Age | Patient age | years | 0-120 |
| Weight | Patient weight | kg | 20-200 |
| Creatinine | Serum creatinine | μmol/L | Male <120, Female <110 |
| Sodium | Serum sodium | mmol/L | 135-145 |
| Potassium | Serum potassium | mmol/L | 3.5-5.5 |
| Glucose | Blood glucose | mmol/L | 3.9-6.1 (fasting) |
| Urea | Blood urea | mmol/L | 2.5-8.0 |
| Gender_Male | Gender (binary) | - | 1=Male, 0=Female |

### **Output**
- **Prediction**: 0 (No CKD) or 1 (CKD Stage 5)
- **Probability**: Float (0.0 to 1.0)
- **Risk Level**: Low / Medium / High

### **Important Notes**

⚠️ **Feature Scaling**: The Random Forest model was trained on **unscaled** features. Do NOT apply StandardScaler during prediction.

⚠️ **Feature Format**: Model expects pandas DataFrame with exact column names (not numpy arrays).

---

## 📁 Project Structure

```
Chronic-Kidney-Disease/
├── app.py                      # Streamlit web interface
├── ckd_api.py                  # FastAPI backend server
├── requirements.txt            # Python dependencies
├── start_api_with_llm.sh      # Startup script
├── .env                        # Environment variables (create this)
├── README.md                   # This file
├── LICENSE                     # License information
│
├── models/                     # Trained models
│   ├── random_forest_model.pkl # Main prediction model
│   ├── feature_names.pkl       # Feature column names
│   ├── scaler.pkl              # Not used (kept for reference)
│   └── schema.py               # Data schemas
│
├── notebooks/                  # Jupyter notebooks
│   ├── experiments.ipynb       # Model training & experiments
│   ├── model_beta.ipynb        # Alternative model development
│   ├── data_cleaning.ipynb     # Data preprocessing
│   └── model_iterative_imputation.ipynb
│
├── data/                       # Raw data files (not included in repo)
│   └── [Excel files]
│
└── tests/                      # Test scripts
    ├── test_api.py
    ├── test_scaling_impact.py
    └── test_model_directly.py
```

---

## 💡 Usage Examples

### **Example 1: Using the Streamlit Interface**

1. Navigate to http://localhost:8501
2. Select "Single Patient Prediction" from sidebar
3. Enter patient values
4. Click "🔍 Predict CKD Risk"
5. View results with visualization

### **Example 2: Using cURL**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "weight": 85,
    "creatinine": 400,
    "sodium": 135,
    "potassium": 5.5,
    "glucose": 8.0,
    "urea": 25,
    "gender_male": 0
  }'
```

### **Example 3: Using Python Requests**

```python
import requests

url = "http://localhost:8000/predict"
patient_data = {
    "age": 65.0,
    "weight": 85.0,
    "creatinine": 400.0,
    "sodium": 135.0,
    "potassium": 5.5,
    "glucose": 8.0,
    "urea": 25.0,
    "gender_male": 0
}

response = requests.post(url, json=patient_data)
result = response.json()

print(f"Prediction: {result['status']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

---

## 🔧 Troubleshooting

### **Issue: uvicorn command not found**

**Solution:**
```bash
pip install uvicorn[standard]
```

### **Issue: Model giving same predictions for all inputs**

**Possible Causes:**
1. Model file is corrupted
2. Wrong model file loaded
3. Feature names mismatch

**Solution:** Retrain the model using `notebooks/experiments.ipynb`

### **Issue: Gemini API not working**

**Check:**
1. GEMINI_API_KEY is set in `.env` file
2. API key is valid and not expired
3. Internet connection is active

### **Issue: ModuleNotFoundError**

**Solution:**
```bash
pip install -r requirements.txt
```

### **Issue: Port already in use**

**Solution:**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn ckd_api:app --port 8001
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

### **Development Guidelines**

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

---

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER:**

This system is for **educational and research purposes only**. It is NOT intended for:

❌ Medical diagnosis  
❌ Clinical decision-making  
❌ Patient treatment planning  
❌ Replacing professional medical advice  

**Always consult qualified healthcare professionals** for medical diagnosis and treatment. The predictions provided by this system should be validated by medical experts before any clinical use.

The developers and contributors assume **no responsibility** for any medical decisions made based on this system's output.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Oyenike Olukiran Ph.D.** - Initial work - [GitHub Profile](https://github.com/Aadunni)

---

## 🙏 Acknowledgments

- Hospital data providers for training dataset
- Google Gemini AI for medical assessment capabilities
- scikit-learn community for ML tools
- FastAPI and Streamlit teams for excellent frameworks

---

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Ayo-Cyber/Chronic-Kidney-Disease/issues)
- **Email**: [Your contact email]
- **Documentation**: See `/docs` endpoint when API is running

---

## 📊 Project Status

🟢 **Active Development** - Regular updates and improvements

### **Roadmap**

- [ ] Add support for more CKD stages
- [ ] Implement user authentication
- [ ] Add data persistence (database)
- [ ] Docker containerization
- [ ] Cloud deployment guide
- [ ] Mobile app interface
- [ ] Multi-language support

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐️ on GitHub!

---

**Made with ❤️ for better healthcare through AI**
