# 🧠 Disease Prediction using Machine Learning

This project predicts the **likelihood of various diseases** using patient symptoms and diagnostic information. It leverages popular machine learning classification algorithms and is trained on a public dataset from Kaggle.

---

## ✅ Objective

Predict the **possible disease** a patient might have based on input features like symptoms, blood test results, and other medical data.

---

## 🧰 Tech Stack

- **Python** (Pandas, NumPy)
- **Scikit-learn** (Logistic Regression, Random Forest, SVM)
- **XGBoost**
- **Matplotlib / Seaborn**
- **Google Colab** / VS Code
- *(Optional)*: Streamlit / Gradio for Web App

---

## 📁 Dataset

- **Source**: [Kaggle Dataset - Disease Prediction using ML](https://www.kaggle.com/datasets)
- Format: `.csv`
- Preprocessing: Handling missing values (NaNs), encoding categorical features, standardizing numerical features

---

## 🔍 Features Used

- Multiple symptoms (fever, headache, nausea, etc.)
- Blood work indicators
- Vital signs (age, weight)
- ~130 feature columns
- Target variable: `prognosis` (disease name)

---

## 📊 Algorithms Compared

| Algorithm            | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.XX     |
| SVM                  | 0.XX     |
| Random Forest        | 0.XX     |
| XGBoost              | 0.XX     |

> 📌 *XGBoost and Random Forest gave the best results in terms of accuracy and generalization.*

---

## 📈 Evaluation Metrics

- Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

---

## 🧪 Sample Prediction

```python
# Input: Symptom features from a patient
sample = X_test.iloc[[0]]
model.predict(sample)
# Output: 'Typhoid'
