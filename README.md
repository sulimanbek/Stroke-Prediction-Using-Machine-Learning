# Stroke Prediction Using Machine Learning 🧠🏥

This project applies **machine learning techniques** to predict the likelihood of a stroke based on patient health records.  
The goal is to support **early risk detection** in healthcare using data-driven models.

The project was developed as part of an Artificial Intelligence course and focuses on **model performance, feature engineering, and medical insight extraction**.

---

## 📌 Project Overview

- Dataset: ~20,000 patient records (healthcare data)
- Target: Binary stroke prediction (stroke / no stroke)
- Challenge: **Highly imbalanced dataset**
- Outcome: Achieved **ROC-AUC score of 0.888**

Key risk factors identified:
- Age
- BMI
- Average glucose level
- Hypertension
- Heart disease

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Naïve Bayes
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Classification (SVC)
- **LightGBM**
- **XGBoost**
- Ensemble modeling (optimized)

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Handling missing values
- One-hot encoding for categorical variables
- Feature binning for visualization
- Class imbalance handling via **upsampling**

### 2️⃣ Exploratory Data Analysis (EDA)
- Stroke distribution analysis
- Feature vs stroke visualizations
- KDE plots for age, BMI, glucose level
- Gender & lifestyle factor analysis

### 3️⃣ Feature Engineering
Custom features created:
- `age / bmi`
- `age * bmi`
- `bmi / prime`
- `obesity`
- `blood_heart`

### 4️⃣ Modeling & Evaluation
- Hyperparameter tuning
- ROC-AUC as the main evaluation metric
- Ensemble model for final prediction

---

## 📊 Results

| Model | ROC-AUC |
|------|--------|
| Baseline LightGBM | ~0.84 |
| Optimized Ensemble | **0.888** |

Feature importance analysis showed **age, BMI, and glucose level** as the most influential predictors.

---

## 📁 Project Structure
├── notebook.ipynb # Full EDA, preprocessing, modeling
├── report.pdf # Academic report (methodology & results)
├── data/ # Dataset (Kaggle source)
├── figures/ # EDA plots and visualizations
└── README.md
