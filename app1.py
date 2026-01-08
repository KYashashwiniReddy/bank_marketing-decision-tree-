import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Bank Marketing – Decision Tree",
    page_icon="🌳",
    layout="centered"
)

st.title("Bank Marketing – Decision Tree Model")
st.caption("Inputs used: Age, Job, Balance, Loan, Contact")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("bank_marketing.csv")

st.subheader("Sample Dataset")
st.dataframe(df.head())

# ---------------- OUTLIER REMOVAL (IQR METHOD) ----------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply outlier removal
df = remove_outliers_iqr(df, 'age')
df = remove_outliers_iqr(df, 'balance')

st.success("Outliers removed using IQR (age & balance)")

# ---------------- ENCODING ----------------
le = LabelEncoder()

df['job'] = le.fit_transform(df['job'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['deposit'] = le.fit_transform(df['deposit'])

# ---------------- INPUT (X) & OUTPUT (y) ----------------
X = df[['age', 'job', 'balance', 'loan', 'contact']]
y = df['deposit']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- DECISION TREE MODEL ----------------
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_leaf=40,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy * 100:.2f}%")

st.text("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------- DECISION TREE VISUALIZATION ----------------
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

# ---------------- UNSEEN DATA PREDICTION ----------------
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

    prediction = model.predict(new_customer)[0]

    if prediction == 1:
        st.success("Customer is likely to SUBSCRIBE")
    else:
        st.error("Customer is NOT likely to subscribe")

