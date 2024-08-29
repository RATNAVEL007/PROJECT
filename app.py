
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import base64

st.set_page_config(page_title="Cirrhosis Disease Diagnosis", page_icon="🩺", layout="wide")

def add_bg_from_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .stApp > div:nth-of-type(1) > div:not(.stSidebar) {{
            color: black;
        }}

        .stSidebar, .stSidebar p, .stSidebar label {{
            color: black !important;
        }}

        .stTextInput, .stButton, .stTimeInput {{
            color: black !important;
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: black !important;
        }}

        div.stButton > button:first-child {{
            background-color: white;
            color: black;
            border: 2px solid black;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
        }}
    .custom-result-box {{
        background-color: white;
        color: black;
        border: 2px solid black;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

#add_bg_from_base64("51ELJDSS+BL._AC_UF894,1000_QL80_.jpg")

@st.cache_data
def load_data():
    return pd.read_csv('cirrhosis.csv')

data = load_data()
X = data.drop('Status', axis=1)  # Using 'Status' as the target column
y = data['Status']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

@st.cache_resource
def load_pipeline_and_model():
    pipeline = joblib.load('cirrhosis_pipeline.pkl')  # Load the preprocessing pipeline
    model = load_model('cirrhosis_disease_model.keras')  # Load the trained model
    return pipeline, model

pipeline, model = load_pipeline_and_model()

st.title("🩺 Cirrhosis Disease Diagnosis Prediction")
st.markdown("---")

st.sidebar.header("🛠️ Feature Selection")
input_data = {}
for feature in X.columns:
    unique_values = X[feature].unique().tolist()
    if len(unique_values) <= 10:
        input_data[feature] = st.sidebar.radio(f"🌿 {feature.replace('_', ' ').title()}:", unique_values)
    else:
        input_data[feature] = st.sidebar.selectbox(f"🌿 {feature.replace('_', ' ').title()}:", unique_values)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✨ Selected Features")
    feature_df = pd.DataFrame([input_data])
    st.dataframe(feature_df.T, use_container_width=True)

with col2:
    st.subheader("🩺 Make a Prediction")

    if st.button("🔍 Predict Disease", key="predict_button"):
        with st.spinner("Analyzing..."):
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=X.columns)
            if input_df.isnull().values.any():
                st.warning("Input data contains NaN values. Please check your inputs.")
                st.stop()
            for column in input_df.columns:
                if input_df[column].dtype == 'object':
                    input_df[column] = input_df[column].astype(str)

            try:
                input_processed = pipeline.transform(input_df)
                prediction = model.predict(input_processed)
                predicted_class_index = np.argmax(prediction)
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                st.success(f"🌟 **Predicted Status:** {predicted_class.upper()} 🎯")
                confidence_scores = prediction[0]
                fig, ax = plt.subplots()
                ax.bar(label_encoder.classes_, confidence_scores, color='#1a76ff')
                ax.set_xlabel('Status Classes')
                ax.set_ylabel('Confidence')
                ax.set_title('Prediction Confidence Scores')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=90)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in transforming input data: {e}")
                st.stop()

st.markdown("---")
st.subheader("🔍 About This Tool")
st.write("""
🩺 This tool uses a deep learning model to predict cirrhosis disease status based on various patient characteristics. Choose the features in the sidebar and click '🔍 Predict Disease' to get a diagnosis. The confidence scores show how confident the model is about each possible status class.
""")

st.markdown("---")
st.markdown("Created by RATNAVEL")

