❤️ Heart Disease Prediction – ML Assignment 2
Course Details

Program: M.Tech (AIML / DSE)

Course: Machine Learning

Assignment: Assignment – 2

Marks: 15

Deployment Platform: Streamlit Community Cloud

📌 Problem Statement

The objective of this project is to build, evaluate, and deploy multiple machine learning classification models to predict the presence of heart disease in patients based on clinical and demographic features.

This assignment demonstrates an end-to-end machine learning workflow, including:

Dataset selection and preprocessing

Implementation of multiple classification algorithms

Performance evaluation using standard metrics

Deployment of an interactive web application using Streamlit

📊 Dataset Description [1 Mark]

Dataset Name: Heart Disease Dataset

File Used: heart.csv

Source: Public dataset (Kaggle / UCI Repository)

Problem Type: Binary Classification

Dataset Characteristics

Number of Instances: 1025

Number of Features: 13

Target Variable: target

1 → Presence of heart disease

0 → No heart disease

Key Features

Age

Sex

Chest pain type

Resting blood pressure

Serum cholesterol

Fasting blood sugar

Resting ECG

Maximum heart rate

Exercise-induced angina

ST depression

Slope of ST segment

Number of major vessels

Thalassemia

This dataset satisfies the minimum requirement of 12 features and 500 instances.

🤖 Models Implemented [6 Marks]

All models were trained and evaluated on the same dataset.

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

📈 Evaluation Metrics Used

Each model was evaluated using the following metrics:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

📋 Model Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.86	0.91	0.85	0.87	0.86	0.72
Decision Tree	0.81	0.83	0.80	0.82	0.81	0.62
KNN	0.84	0.88	0.83	0.85	0.84	0.68
Naive Bayes	0.82	0.87	0.81	0.83	0.82	0.64
Random Forest	0.89	0.94	0.88	0.90	0.89	0.78
XGBoost	0.91	0.96	0.90	0.92	0.91	0.82
🧠 Observations on Model Performance [3 Marks]
ML Model	Observation
Logistic Regression	Performs well as a baseline model with good interpretability and stable performance.
Decision Tree	Simple and interpretable but prone to overfitting, leading to lower generalization.
KNN	Effectively captures local patterns but is sensitive to feature scaling and choice of K.
Naive Bayes	Fast and efficient; performs reasonably well despite strong independence assumptions.
Random Forest	Achieves high performance by reducing variance through ensemble learning.
XGBoost	Best performing model with highest accuracy and MCC due to advanced boosting and regularization.
🌐 Streamlit Application Features [4 Marks]

The deployed Streamlit application includes:

📁 Dataset Upload (CSV – test data only)

🔽 Model Selection Dropdown

📊 Display of Evaluation Metrics

🔍 Confusion Matrix Visualization

📄 Classification Report Display

🗂️ Project Structure
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   ├── logistic_model.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl

⚙️ Installation & Execution
Install Dependencies
pip install -r requirements.txt

Run the Application
streamlit run app.py

📦 requirements.txt
streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn
xgboost

🧪 Execution Environment

Assignment executed on BITS Virtual Lab

Screenshot of execution has been included in the final PDF submission

📜 Academic Integrity Declaration

This assignment has been independently implemented in accordance with the Academic Integrity Guidelines.
AI tools were used only for learning support, and no direct copy-paste submissions were made.

✅ Final Submission Checklist

✔ GitHub repository link working

✔ Streamlit app link opens correctly

✔ All six models implemented

✔ Required metrics computed

✔ README.md included in PDF

✔ BITS Virtual Lab screenshot attached