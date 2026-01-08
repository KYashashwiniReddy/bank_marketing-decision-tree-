import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Marketing – Decision Tree",
    page_icon="🌳",
    layout="centered"
)

st.title("Bank Marketing – Decision Tree App")

# ---------------- FILE UPLOAD ----------------
df = pd.read_csv("bank_marketing.csv")
# ---------------- SAMPLE DATA ----------------
st.subheader("📄 Sample Dataset")
st.dataframe(df.head())

# ---------------- OUTLIER HANDLING (CAPPING) ----------------
def cap_outliers(df, column):
    lower = df[column].quantile(0.05)
    upper = df[column].quantile(0.95)
    df[column] = df[column].clip(lower, upper)
    return df

# Apply outlier handling
df = cap_outliers(df, 'age')
df = cap_outliers(df, 'balance')

st.success("Outliers handled using capping (5%–95%)")

# ---------------- ENCODING ----------------
le = LabelEncoder()

df['job'] = le.fit_transform(df['job'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['deposit'] = le.fit_transform(df['deposit'])

# ---------------- FEATURES & TARGET ----------------
X = df[['age', 'job', 'balance', 'loan', 'contact']]
y = df['deposit']

# ---------------- TRAIN–TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- DECISION TREE MODEL ----------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------
y_pred = model.predict(X_test)

st.subheader("Model Performance")

st.metric("Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

st.text("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------- TREE VISUALIZATION ----------------
st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Not Subscribe', 'Subscribe'],
    filled=True,
    ax=ax
)
st.pyplot(fig)

# ---------------- PREDICTION ON UNSEEN DATA ----------------
st.subheader("Predict for New Customer")

age = st.number_input("Age", 18, 100, 40)
job = st.number_input("Job (Encoded)", 0, 20, 2)
balance = st.number_input("Balance", -5000, 100000, 1500)
loan = st.number_input("Loan (0 = No, 1 = Yes)", 0, 1, 0)
contact = st.number_input("Contact (Encoded)", 0, 5, 1)

if st.button("Predict"):
    new_customer = pd.DataFrame(
        [[age, job, balance, loan, contact]],
        columns=X.columns
    )

    new_customer_scaled = scaler.transform(new_customer)
    prediction = model.predict(new_customer_scaled)[0]

    if prediction == 1:
        st.success("Customer is likely to SUBSCRIBE")
    else:
        st.error("Customer is NOT likely to subscribe")
