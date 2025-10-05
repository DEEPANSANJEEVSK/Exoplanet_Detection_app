# ================================================
# 🔭 AI-Based Exoplanet Detection (Kepler Mission)

# ================================================

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import os

# --- Streamlit UI Setup ---
st.title("🚀 Exoplanet Detection by phoneix ROX ")
st.markdown("""
This project uses **AI and Machine Learning** to detect potential exoplanets from NASA Kepler Mission Data.  
Upload your dataset or use the default Kepler data.
""")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("nasa", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    default_file = "nasa.csv"
    if os.path.exists(default_file):
        st.info("No file uploaded — using default 'nasa.csv'.")
        data = pd.read_csv(default_file)
    else:
        st.error(f"❌ Default file '{default_file}' not found. Please upload a CSV file.")
        st.stop()

# --- Display Data Overview ---
st.subheader("📊 Data Overview")
st.dataframe(data.head())
st.write("Dataset shape:", data.shape)

# --- Preprocess Dataset ---
if 'koi_disposition' not in data.columns:
    st.error("Dataset missing required column 'koi_disposition'. Please check your file.")
    st.stop()

# Keep only CONFIRMED or FALSE POSITIVE
data = data[data['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
if data.shape[0] == 0:
    st.error("❌ No valid rows after filtering. Dataset must contain CONFIRMED or FALSE POSITIVE entries.")
    st.stop()

# Encode labels: CONFIRMED → 1, FALSE POSITIVE → 0
data['label'] = data['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# Drop unnecessary columns
for col in ['kepid', 'koi_disposition']:
    if col in data.columns:
        data = data.drop(col, axis=1)

# --- Separate Features and Labels ---
X = data.drop('label', axis=1)
y = data['label']

# --- Drop Non-Numeric Columns ---
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    st.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(non_numeric_cols, axis=1)

if X.shape[1] == 0:
    st.error("❌ No numeric features left after preprocessing. Cannot train model.")
    st.stop()

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("📈 Model Evaluation Results")
st.write(f"✅ **Accuracy:** {acc*100:.2f}%")
st.text(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig, clear_figure=True)

# --- Feature Importance ---
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.subheader("🌟 Feature Importance (Top 10)")
st.bar_chart(feat_imp.head(10))

# --- Custom Prediction Input ---
st.subheader("🧮 Try a Custom Prediction")

sample = []
selected_features = st.multiselect(
    "Select features for prediction (at least 1)", 
    options=list(X.columns), 
    default=list(X.columns[:44])
)

if selected_features:
    for col in selected_features:
        val = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
        sample.append(val)

    if st.button("🔍 Predict Exoplanet?"):
        prediction = model.predict([sample])[0]
        if prediction == 1:
            st.success("✅ This object is likely an **Exoplanet!** 🌍")
        else:
            st.error("❌ This object is likely **Not a Planet.**")
else:
    st.warning("⚠️ Please select at least one feature for prediction.")

st.caption("Developed by phoneix ROX | Powered by Python, Scikit-learn & Streamlit")

