# ❤️ Heart Disease Prediction – ML Assignment 2

## Course Information
- **Program:** M.Tech (AIML / DSE)  
- **Course:** Machine Learning  
- **Assignment:** Assignment – 2  
- **Marks:** 15  

---

## a. Problem Statement

The objective of this assignment is to design, evaluate, and deploy multiple machine
learning classification models to predict the presence of heart disease in patients
using clinical and demographic features.

The project demonstrates a complete **end-to-end machine learning workflow**:
- Dataset preprocessing and cleansing  
- Implementation of multiple classification models  
- Evaluation using standard performance metrics  
- Deployment of an interactive Streamlit web application  

---

## b. Dataset Description  **[1 Mark]**

- **Dataset Name:** Heart Disease Dataset  
- **File Used:** `heart.csv`  
- **Problem Type:** Binary Classification  

### Dataset Characteristics
- **Number of Instances:** ~1025  
- **Number of Features:** 13  
- **Target Variable:** `target`  
  - `1` → Presence of heart disease  
  - `0` → Absence of heart disease  

### Data Preprocessing & Cleansing
As implemented in `app.py`, the following data cleansing steps were applied **before
model training**:
- Removal of duplicate records  
- Handling missing values using **median imputation**  
- Outlier handling using the **Interquartile Range (IQR) capping method**  
- Feature scaling using **StandardScaler**  

These steps ensure realistic model performance and prevent misleadingly high accuracy.

---

## c. Models Used & Evaluation Metrics  **[6 Marks]**

All models were trained and evaluated on the **same cleansed dataset**.

### Implemented Models
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

### Evaluation Metrics
Each model was evaluated using:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

## 📊 Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | ~0.85 | ~0.90 | ~0.84 | ~0.86 | ~0.85 | ~0.70 |
| Decision Tree | ~0.80 | ~0.84 | ~0.79 | ~0.81 | ~0.80 | ~0.60 |
| KNN | ~0.83 | ~0.87 | ~0.82 | ~0.84 | ~0.83 | ~0.66 |
| Naive Bayes | ~0.82 | ~0.86 | ~0.81 | ~0.83 | ~0.82 | ~0.64 |
| Random Forest | ~0.89 | ~0.94 | ~0.88 | ~0.90 | ~0.89 | ~0.78 |
| XGBoost | ~0.91 | ~0.95 | ~0.90 | ~0.92 | ~0.91 | ~0.82 |

*(Exact values may vary slightly due to random train-test split.)*

---

## 🧠 Observations on Model Performance  **[3 Marks]**

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Provides a strong baseline with good generalization after data cleansing. |
| Decision Tree | Easy to interpret but shows reduced performance due to depth restriction to prevent overfitting. |
| KNN | Performs reasonably well after feature scaling but is sensitive to the choice of K. |
| Naive Bayes | Fast and efficient; performs well despite independence assumptions. |
| Random Forest | Achieves high accuracy by combining multiple decision trees while controlling variance. |
| XGBoost | Best performing model due to gradient boosting and regularization, even after complexity control. |

---

## 🌐 Streamlit Application Features

The Streamlit application includes:
- CSV dataset upload option  
- Model selection dropdown  
- Display of all required evaluation metrics  
- Confusion matrix visualization  
- Classification report display  

---

## 🗂️ Project Structure

project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    ├──   logistic_model.py
    ├──   decision_tree.py
    ├──   knn.py
    ├──   naive_bayes.py
    ├──   random_forest.py
    └──   xgboost.py

## ⚙️ How to Run the Application

### Install Dependencies
pip install -r requirements.txt
### Run Streamlit App
streamlit run app.py
## 🧪 Execution Environment

- The assignment was executed on **BITS Virtual Lab**
- A screenshot of execution has been included in the final PDF submission

---

## 📜 Academic Integrity Declaration

This assignment has been independently implemented in accordance with the
Academic Integrity Guidelines. AI tools were used only for conceptual understanding
and learning support, and not for direct copy-paste submissions.

---

## ✅ Final Submission Checklist

- GitHub repository link works  
- Streamlit app link opens correctly  
- All six models implemented  
- Data cleansing applied before modeling  
- Evaluation metrics displayed  
- README.md included in submitted PDF  
- BITS Virtual Lab screenshot attached 
