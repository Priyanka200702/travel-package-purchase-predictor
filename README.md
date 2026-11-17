# Wellness Tourism Package Purchase Prediction

A machine learning classification web application that predicts whether a customer will purchase a wellness tourism package (`ProdTaken`) based on demographic details, travel history, and sales interaction metrics.

## Overview

Acquiring new customers in the travel industry is costly when sales teams reach out to leads indiscriminately. This project analyzes customer records from the **Travel.csv** dataset (4,888 rows, 20 features) to identify high-intent buyers, helping marketing and sales teams prioritize follow-ups and improve conversion rates.

## Key Features & Pipeline

1. **Data Cleaning & Imputation**
   - Standardized inconsistent categorical labels (`'Fe Male'` → `'Female'`, `'Single'` → `'Unmarried'`).
   - Imputed missing values using median for skewed numerical columns (`Age`, `NumberOfTrips`, `MonthlyIncome`), rounded mean for `DurationOfPitch`, mode for categorical/discrete columns (`TypeofContact`, `NumberOfFollowups`, `PreferredPropertyStar`), and `0` for `NumberOfChildrenVisiting`.
   - Dropped non-predictive identifier (`CustomerID`).
   - Changed the datatypes of 'NumberOfFollowups','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome'
2. **Feature Engineering**
   - Combined `NumberOfPersonVisiting` and `NumberOfChildrenVisiting` into a single `TotalVisiting` feature.
3. **Preprocessing (`ColumnTransformer`)**
   - Applied `OneHotEncoder(drop='first')` to categorical features (`TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation`).
   - Applied `StandardScaler()` to numerical features.
   - Bundled preprocessing and the estimator into a single scikit-learn `Pipeline` (`travel_Model.joblib`) to prevent data leakage and simplify deployment.
4. **Hyperparameter Tuning & Model Evaluation**
   - Evaluated Decision Tree and Random Forest classifiers and tuned hyperparameters using `RandomizedSearchCV` (3-fold cross-validation across 100 candidates).

## Model Performance (Test Set)

Tuned **Random Forest Classifier** (`n_estimators=1000`, `max_features=8`, `min_samples_split=2`, `max_depth=None`):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **93.76%** |
| **Weighted F1-Score** | **0.9338** |
| **Precision** | **0.9643** |
| **Recall** | **0.7068** |
| **ROC-AUC Score** | **0.8502** |

## Project Structure

```text
├── Travel.csv                 # Raw dataset
├── travelPack.ipynb           # EDA, preprocessing, model training, and tuning
├── travel_Model.joblib        # Saved scikit-learn Pipeline (ColumnTransformer + Random Forest)
├── app.py                     # Streamlit web application for live predictions
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Priyanka200702/travel-package-purchase-predictor.git
cd travel-package-purchase-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Core libraries used: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `seaborn`, `matplotlib`)*

### 3. Run the Web Application
```bash
streamlit run app.py
```
