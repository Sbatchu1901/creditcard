# 💳 Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions. It demonstrates data preprocessing, SMOTE oversampling, model training (Logistic Regression and Random Forest), evaluation, and optional SHAP explainability.

---

## 📁 Dataset

The dataset used is from [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
**Note:** The CSV file (`creditcard.csv`) is not included in this repo due to GitHub’s 100MB file limit. Please download it manually from Kaggle and place it in the root folder.

---

## ⚙️ Features

- ✅ Data cleaning and scaling
- ✅ SMOTE for class imbalance
- ✅ Logistic Regression & Random Forest
- ✅ Confusion matrix, ROC AUC score, classification report
- ✅ Optional: SHAP explainability for model transparency

---

## 🛠️ Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/sbatchu1901/creditcard.git
cd creditcard
pip install -r requirements.txt
📊 Model Performance
Model	Accuracy	Precision (Fraud)	Recall (Fraud)	ROC AUC
Logistic Regression	~97%	Low	High (~88%)	~0.96
Random Forest	~100%	93%	76%	~0.98
