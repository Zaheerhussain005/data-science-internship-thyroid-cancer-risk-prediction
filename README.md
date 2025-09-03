# 🧬 Thyroid Cancer Risk Prediction with Machine Learning  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)](https://scikit-learn.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Healthcare AI](https://img.shields.io/badge/AI-Healthcare-red)](#)  

---

## 📖 Overview
This project implements **machine learning models** to predict whether thyroid cancer cases are **benign (non-cancerous)** or **malignant (cancerous)** using clinical and demographic data.  

We evaluated multiple models including:  
- **Logistic Regression**  
- **Random Forest**  
- **Gradient Boosting (GBM)**  
- **AdaBoost**  
- **Decision Tree**  
- **Naive Bayes**  
- **XGBoost & LightGBM**  

Ensemble models (e.g., **Gradient Boosting, AdaBoost**) achieved the **highest accuracy (~82.5%)**, making them the most promising candidates for real-world clinical decision support.  

---

## 🚨 Why This Matters
- Thyroid cancer is one of the fastest-growing endocrine cancers.  
- Early prediction of malignancy and recurrence helps:  
  - 🧪 Detect high-risk patients earlier  
  - 🎯 Enable personalized treatment strategies  
  - 💡 Reduce unnecessary surgeries or procedures  
- Traditional diagnostic methods are invasive, time-consuming, and error-prone.  
- Machine learning offers a **data-driven, non-invasive, and scalable solution**.  

---

## 🧾 Dataset
- **Size**: 212,691 anonymized patient records  
- **Features**:  
  - `Age` – Risk varies by age group  
  - `Gender` – Thyroid disorders are gender-influenced  
  - `Family History` – Genetic predisposition  
  - `TSH (Thyroid Stimulating Hormone)`  
  - `T3, T4 levels` – Thyroid hormone indicators  
  - `Diagnosis (Target)` – 0 = Benign, 1 = Malignant  

➡️ Data was cleaned, normalized, and encoded for ML processing.  

---

## ⚙️ Preprocessing Pipeline
1. **Missing Values** → Median for numeric, mode for categorical  
2. **Encoding** → Label Encoding & One-Hot Encoding  
3. **Scaling** → StandardScaler for numerical features  
4. **Feature Selection** → Removed redundant/non-informative columns  
5. **Train-Test Split** → 80/20 stratified split to preserve class balance  

---

## 🛠️ Model Implementation
Each model was trained using **scikit-learn** and **gradient boosting frameworks (XGBoost, LightGBM)**.  

- Logistic Regression → Baseline, interpretable  
- Random Forest → Robust tree ensemble  
- Gradient Boosting & AdaBoost → Best performers (82.5% accuracy)  
- Decision Tree → Weaker (70.2%), prone to overfitting  
- Naive Bayes → Fast, probabilistic benchmark  

---

## 📊 Results
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 82.5%    | Balanced  | Balanced | Strong |
| Gradient Boosting    | 82.5%    | High      | High     | Strong |
| AdaBoost             | 82.5%    | Good      | Good     | Strong |
| Random Forest        | ~80%     | Moderate  | Moderate | Good |
| Decision Tree        | 70.2%    | Low       | Low      | Weak |
| Naive Bayes          | ~75%     | Moderate  | Moderate | Fair |

✅ **Top models: Logistic Regression, Gradient Boosting, AdaBoost**  
❌ **Weak model: Decision Tree (overfit)**  

---

## 🧑‍⚕️ Clinical Implications
- Early risk prediction aids **doctors** in proactive diagnosis.  
- Can be integrated into **Electronic Health Records (EHR)** for real-time alerts.  
- Models highlight key clinical features:  
  - High **TSH** levels  
  - Abnormal **T3/T4 ratios**  
  - Family history & age  

⚠️ **Note:** These models are decision-support tools — not replacements for clinical judgment.  

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/thyroid-cancer-prediction.git
cd thyroid-cancer-prediction


###2. Install Requirements

-pip install -r requirements.txt


###3. Train a Model

-python train.py --model logistic_regression --input data/thyroid_dataset.csv

###📂 Project Structure

```├── data/               # Thyroid cancer dataset
├── notebooks/          # Jupyter notebooks for experiments
├── models/             # Saved models (LogReg, RF, GBM, etc.)
├── results/            # Accuracy reports, confusion matrices
├── src/                # Core Python scripts
├── requirements.txt    # Dependencies
└── README.md           # You are here
```

###🔮 Future Work


🧑‍💻 Author
- Zaheer Hussain – MSc Data Science, NUST MISIS (Moscow)
