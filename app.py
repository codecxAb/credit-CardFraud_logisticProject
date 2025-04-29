import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    return df

# Cache the scaling transformations
@st.cache_data
def get_scalers(df):
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    df['Amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    return df, scaler_amount, scaler_time

# Cache the SMOTE preprocessing
@st.cache_data
def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Reduce synthetic samples
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res, X_test, y_test

# Cache the model training
@st.cache_resource
def train_model(X_res, y_res):
    model = LogisticRegression(
        max_iter=1000,  # To ensure convergence
        class_weight='balanced',  # To handle imbalanced data
    )
    model.fit(X_res, y_res)
    return model

# Load and preprocess the dataset
df = load_data()
df, scaler_amount, scaler_time = get_scalers(df)

# Preprocess the data and split into training and testing sets
X_res, y_res, X_test, y_test = preprocess_data(df)

# Train the model
model = train_model(X_res, y_res)

# Evaluation metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])
auc_score = roc_auc_score(y_test, y_prob)
fraud_precision = precision_score(y_test, y_pred, pos_label=1)

# Streamlit UI setup
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud risk:")

# User input for Amount and Time
amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
time_of_day = st.selectbox(
    "When did the transaction happen?",
    ("Morning (9AM-11AM)", "Afternoon (12PM-4PM)", "Evening (5PM-8PM)", "Night (9PM-11PM)", "Late Night (12AM-5AM)")
)

# Map user selection of time of day to seconds
if time_of_day == "Morning (9AM-11AM)":
    time_seconds = np.random.randint(10000, 30000)
elif time_of_day == "Afternoon (12PM-4PM)":
    time_seconds = np.random.randint(30000, 60000)
elif time_of_day == "Evening (5PM-8PM)":
    time_seconds = np.random.randint(60000, 90000)
elif time_of_day == "Night (9PM-11PM)":
    time_seconds = np.random.randint(90000, 120000)
else:
    time_seconds = np.random.randint(120000, 170000)

# Function to combine all PCA features with user input
def generate_input_features(amount, time_seconds):
    legit_samples = df[df['Class'] == 0].drop('Class', axis=1)
    random_row = legit_samples.sample(n=1, random_state=np.random.randint(1000))
    random_row['Amount'] = scaler_amount.transform(np.array(amount).reshape(-1, 1))[0]
    random_row['Time'] = scaler_time.transform(np.array(time_seconds).reshape(-1, 1))[0]
    return random_row.values.flatten()

# Prediction
if st.button("Predict"):
    input_features = np.array([generate_input_features(amount, time_seconds)])
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    # Adjust threshold for fraud detection to improve precision
    fraud_threshold = 0.7  # Higher threshold for fraud classification
    if probability >= fraud_threshold:
        prediction = 1  # Predict fraud
    else:
        prediction = 0  # Predict legitimate

    # Show result
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Risk Score: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {(1-probability):.2%})")

    # Visual probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        },
        title={'text': "Fraud Risk (%)"}
    ))
    st.plotly_chart(fig)

# Sidebar metrics
with st.sidebar:
    st.header("Model Metrics")
    st.subheader("Classification Report")
    st.text(report)
    st.subheader("ROC AUC Score")
    st.write(f"{auc_score:.4f}")
