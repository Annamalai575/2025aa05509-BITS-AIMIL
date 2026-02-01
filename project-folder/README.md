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
| Logistic Regression | 0.829 | 0.89 | 0.818 | 0.878 | 0.847 | 0.656 |
| Decision Tree | 0.75 | 0.827 | 0.789 | 0.732 | 0.759 | 0.502 |
| KNN | 0.803 | 0.864 | 0.81 | 0.829 | 0.819 | 0.602 |
| Naive Bayes | 0.816 | 0.891 | 0.846 | 0.805 | 0.825 | 0.632 |
| Random Forest | 0.776 | 0.856 | 0.786 | 0.805 | 0.795 | 0.549 |
| XGBoost | 0.763 |0.863 | 0.767 | 0.805 | 0.786 | 0.522 |


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

**Deployment Details**
**Platform used = Streamlit Community Cloud
Python Version used = 3.10
Deployment Type = Free Tier
GitHub Repository = https://github.com/Annamalai575/2025aa05509-BITS-AIMIL/tree/main/project-folder
Live Application URL = https://2025aa05509-bits-aimil-gzspr2hjgibbzsjxgaypmq.streamlit.app/**


## ✅ Final Submission Checklist

- GitHub repository link works  
- Streamlit app link opens correctly  
- All six models implemented  
- Data cleansing applied before modeling  
- Evaluation metrics displayed  
- README.md included in submitted PDF  
- BITS Virtual Lab screenshot attached 


