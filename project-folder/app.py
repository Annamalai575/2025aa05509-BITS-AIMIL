import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("### ML Assignment 2 – Cleansed & Realistic Evaluation")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload Heart Disease CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload heart.csv to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# DATA CLEANSING (IMPORTANT PART)
# --------------------------------------------------
st.subheader("🧹 Data Cleansing Steps Applied")

# 1. Remove duplicates
initial_rows = df.shape[0]
df = df.drop_duplicates()
st.write(f"✔ Removed duplicates: {initial_rows - df.shape[0]} rows")

# 2. Handle missing values
imputer = SimpleImputer(strategy="median")
df[df.columns] = imputer.fit_transform(df)
st.write("✔ Missing values handled using median strategy")

# 3. Outlier capping (IQR method)
def cap_outliers(data):
    capped = data.copy()
    for col in capped.columns:
        if col != "target":
            Q1 = capped[col].quantile(0.25)
            Q3 = capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            capped[col] = np.clip(capped[col], lower, upper)
    return capped

df = cap_outliers(df)
st.write("✔ Outliers capped using IQR method")

# --------------------------------------------------
# FEATURE / TARGET SPLIT
# --------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# MODELS (COMPLEXITY CONTROLLED)
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

model = models[model_name]

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
if model_name in ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
else:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
st.subheader("📊 Model Performance (Post-Cleansing)")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred)
}

c1, c2, c3 = st.columns(3)
items = list(metrics.items())

for col, pair in zip([c1, c2, c3], [items[:2], items[2:4], items[4:]]):
    with col:
        for name, val in pair:
            st.metric(name, round(val, 3))

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("🧩 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.subheader("📄 Classification Report")
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True)
).transpose()
st.dataframe(report_df)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
# st.markdown("---")
# st.markdown(
#     "✔ Dataset cleansed before modeling  \n"
#     "✔ Outliers handled, complexity controlled  \n"
#     "✔ Metrics now reflect realistic ML performance"
# )
