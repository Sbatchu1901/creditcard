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
'''
---
##
### '''
 📊 Model Performance

| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | ROC AUC |
|--------------------|----------|-------------------|----------------|---------|
| Logistic Regression| ~97%     | Low               | ~88%           | ~0.96   |
| Random Forest      | ~100%    | 93%               | 76%            | ~0.98   |

=
CreditCardFraudDetection/
├── cc.py                  # Data loading
├── preprocessing.py       # Preprocessing + SMOTE
├── modeltraining.py       # ML models
├── evaluation.py          # Metrics & plots
├── mainn.py               # Main pipeline
├── shap_explainer.py      # SHAP analysis (optional)
├── .gitignore
├── requirements.txt
└── README.md
----------------------
### 🙌 Acknowledgments
Kaggle Dataset

scikit-learn, imbalanced-learn, matplotlib, shap
------------------
### Why This Matters
This project tackles real-world class imbalance and demonstrates practical fraud detection — a core challenge in financial institutions like Fiserv.

It showcases:

Responsible ML pipeline

Strong model evaluation

Explainable AI skills
------------------
### Contact
Made by @Sbatchu1901
Feel free to connect or reach out!
