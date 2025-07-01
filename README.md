# 🩺 Chronic Kidney Disease (CKD) Stage Prediction Model

## 🔍 Project Overview

This project is focused on building a machine learning pipeline to **predict the stage of Chronic Kidney Disease (CKD)** based on clinical patient data. The aim is to support early detection and stage-based intervention strategies using structured clinical features.

Patient records were sourced from **two different hospitals**, merged, cleaned, and used to train several classification models capable of predicting **six CKD stages** — from no CKD (Stage 0) to advanced CKD (Stage 5).

---

## 🏥 Dataset Description

* **Source**: Two anonymized hospital datasets.
* **Target Variable**: CKD stage (`stage`) represented as:

  * `0` → No CKD
  * `1` → Stage 1
  * `2` → Stage 2
  * `3` → Stage 3
  * `4` → Stage 4
  * `5` → Stage 5
* **Merge Strategy**:

  * Harmonized column names and formats across datasets.
  * Concatenated into a single DataFrame.
  * Resolved missing values and handled duplicate entries.

---

## 📊 Project Workflow

### 1. **Data Collection and Merging**

* Datasets from Hospital A and Hospital B were imported.
* Unified column structures were created.
* Merged datasets using vertical concatenation.
* Verified class distributions across merged records.

### 2. **Data Cleaning & Preprocessing** 

* Missing data filled using:

  * Mean for numerical fields
  * Mode for categorical fields
* Label encoding of categorical variables
* Standardization of feature units and formats
* Duplicate removal and consistency checks

### 3. **Exploratory Data Analysis (EDA)**

* Visualizations of:

  * Feature distributions
  * CKD stage-wise trends
  * Feature correlation heatmaps
* Identified imbalance in stage distributions
* Detected and addressed outliers

### 4. **Feature Engineering & Initial Experiments**

* Selected clinically relevant features based on correlation and EDA.
* Early experiments with Logistic Regression and Decision Trees to validate signal.

### 5. **Model Building & Evaluation** *(Main Notebook)*

* Performed stratified train-test split (to retain CKD stage balance).
* Trained several models including:

  * Decision Tree
  * Random Forest
  * XGBoost
  * CatBoost 
  * SVM 

* Evaluation Metrics:

  * Overall accuracy
  * Per-class precision, recall, F1-score
  * Confusion matrix
  * Classification report
* Tracked comparative model performance
* Visualized feature importance for interpretability

---

## 📈 Results Summary

* Multiclass classification metrics were reported per CKD stage.
* CatBoost and XGBoost showed strong performance on both precision and generalization.
* Final model chosen based on:

  * F1 macro average
  * Performance on minority CKD stages
  * Interpretability

---

## 📁 Project Structure

```
CKD-Model/
│
├── data/
│   ├── hospital_a.csv
│   ├── hospital_b.csv
│   └── merged_cleaned_dataset.csv
│
├── notebooks/
│   ├── 1_data_cleaning_and_eda.ipynb
│   └── 2_ckd_model_and_metrics.ipynb
│
├── models/
│   └── best_ckd_stage_model.pkl
│
├── visuals/
│   └── feature_importance_plot.png
│
├── src/
│   ├── __init__.py
│   └── main.py               # Main script to run training/inference
│
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

A `requirements.txt` file has been provided and includes the major libraries:

To install:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Tools & Libraries

* Python 3.x
* pandas, numpy
* scikit-learn
* xgboost, catboost
* seaborn, matplotlib

---


## 🤝 Acknowledgements

* This work draws on collaborative efforts between clinical staff of University of Ibadan, University College Hospital and Ladoke Akintola Teaching Hospital Ogbomoso.
* Special thanks to the contributing hospitals for anonymized datasets.

