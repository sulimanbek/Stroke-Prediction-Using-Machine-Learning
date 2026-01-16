# Stroke Rate Prediction using Machine Learning 🧠📊

This project builds an **end-to-end machine learning pipeline** to predict stroke risk using patient healthcare data.  
It focuses on **data preprocessing, exploratory data analysis (EDA), feature engineering, class imbalance handling, and model optimization**.

The final ensemble model achieves an **ROC-AUC score of 0.888**, highlighting the effectiveness of ML in early stroke risk detection.

---

## Problem Statement

Stroke is a life-threatening medical condition where early detection can significantly improve outcomes.  
Using structured healthcare data, this project aims to predict whether a patient is at risk of stroke based on demographic, lifestyle, and medical factors.

---

##  Dataset

- Source: Kaggle Healthcare Stroke Dataset
- Rows: ~20,000 patients
- Features: 12
- Target: `stroke` (0 = No stroke, 1 = Stroke)
- Challenge: **Severely imbalanced dataset**

### Key Features
- Age
- Gender
- Hypertension
- Heart Disease
- BMI
- Average Glucose Level
- Smoking Status
- Work Type
- Residence Type

---

## ⚙️ Project Pipeline

### 1️⃣ Data Preprocessing
- Missing value handling
- One-hot encoding for categorical variables
- Feature binning for analysis
- Dataset splitting (`train.csv`, `test.csv`)
- **Upsampling** of minority class (stroke)

### 2️⃣ Exploratory Data Analysis (EDA)
- Stroke distribution analysis
- Stroke rate by age group
- Gender vs stroke rate
- Hypertension & heart disease impact
- BMI & glucose level visualization
- KDE plots for age distributions

### 3️⃣ Feature Engineering
Custom engineered features:
- `age / bmi`
- `age * bmi`
- `bmi / prime`
- `obesity`
- `blood_heart`

These features improved model performance and interpretability.

### 4️⃣ Modeling
Models explored:
- Logistic Regression
- Naïve Bayes
- Decision Tree
- **Random Forest**
- KNN
- SVM
- **LightGBM**
- **XGBoost**

Final approach:
- Hyperparameter tuning
- **Ensemble modeling**
- Evaluation using ROC-AUC

---

## 📊 Results

| Model | ROC-AUC |
|-----|--------|
| Baseline LightGBM | ~0.84 |
| Optimized Ensemble (LGBM + XGB) | **0.888** |

### Most Influential Features
- Age
- BMI
- Average Glucose Level
- Hypertension
- Heart Disease

---

## 📁 Repository Structure
├── train.csv # Training dataset
├── test.csv # Test dataset
├── sample_submission.csv # Kaggle-style submission format
├── healthcare-dataset-stroke-data.csv
├── Stroke Rate Prediction.ipynb 
└── README.md

How to Run the Script

Ensure you have all the required libraries installed. These include:

numpy
matplotlib
seaborn
sklearn
lightgbm
xgboost
pandas
Load the required datasets. The script expects two datasets:

A training dataset ('./realifeDataSets/train.csv')
A test dataset ('./realifeDataSets/test.csv')
A real world dataset ('./healthcare-dataset-stroke-data.csv')
Run the Python script. You can do this in a Python environment or Jupyter notebook. The script will load the data, preprocess it, perform exploratory data analysis, and then apply and evaluate several machine learning models.
