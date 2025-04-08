
# 🔄 Churn Prediction using Machine Learning

This project aims to predict customer churn for a video streaming service using machine learning techniques. The goal is to identify users likely to cancel their subscription so the business can take proactive retention steps.

---

## 📁 Dataset Overview

- **train.csv**: 243,787 rows of historical customer data with churn labels.
- **test.csv**: 104,480 rows of customer data for which churn predictions are required.

---

## ✅ Approach

### 1. Data Loading
- Loaded training and test datasets using `pandas`.

### 2. Exploratory Data Analysis (EDA)
- Explored data distribution, missing values, class imbalance and churn patterns.
- Visualized key trends in features like monthly charges, account age, and ratings.
- Pairwise correlation graphs and matrix.
- Find high correlation features.

### 3. Data Cleaning
- Filled missing values.
- Dropped irrelevant columns like `CustomerID`.
- Removed high correlation columns.
- Standardize numeric data dimensions (for LR).

### 4. Feature Engineering
- Used `Binary Encoding` for binary features (Yes/No).
- Used `One-Hot Encoding` for multi-category columns (for LR).
- Used 'Label Encoding' for multi-category columns (for tree based models).
- Created custom features like `AvgChargePerMonth` (if needed).

### 5. Modeling
- Trained two separate models:
  - `XGBoostClassifier` (label-encoded features)
  - `LogisticRegression` (one-hot encoded + scaled features)

### 6. Hyperparameter Tuning
- Applied `GridSearchCV` to tune XGBoost parameters for better ROC AUC performance.

### 7. Ensemble
- Combined predictions from both models using weighted average:
  ```python
  final_probs = 0.6 * xgb_probs + 0.4 * lr_probs
  ```

### 8. Submission
- Created `prediction_df` with `CustomerID` and `predicted_probability`.
- Saved output as `submission.csv`.

---

## 📦 Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for ML models, preprocessing, and tuning)
- XGBoost

---

## 📈 Model Evaluation

- Optimized using ROC AUC score with Stratified K-Fold cross-validation.
- Ensemble approach improved generalization by leveraging both tree-based and linear models.

---

## 🚀 Getting Started

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Run the notebook
```bash
jupyter notebook ChurnPrediction.ipynb
```

---

## 📬 Output

Final output: `submission.csv` with columns:
- `CustomerID`
- `predicted_probability` (likelihood of churn between 0 and 1)

---

