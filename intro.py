import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Streamlit App Title
st.title("Water Quality Prediction with XGBoost")

# List of models (Ensure these files exist in your Streamlit app directory)
models = {
    "XGBoost Model": "xgb_model.pkl",
    "Random Forest Model": "rf_model.pkl",
    "Decision Tree Model": "dt_model.pkl",
    "AdaBoost Model": "ab_model.pkl",
    "Multi_Layer Perceptron Model": "mlp_model.pkl",
    "Support Vector Machine Model": "svm_model.pkl",
    "Gradient Boosting Model": "gb_model.pkl"
}

# Dropdown for model selection
selected_model = st.selectbox("Select a Model", list(models.keys()))

# GitHub raw dataset URL (Replace with your actual dataset link)
github_url = "https://github.com/nishifwd/machine-learning/blob/main/Dataset.csv"

if st.button("Load Dataset and Predict"):
    try:
        # Load the selected model
        model_path = models[selected_model]
        loaded_model = joblib.load(model_path)

        # Load dataset from GitHub
        data = pd.read_csv(github_url)

        # Drop unnecessary columns
        data = data.drop(columns=['WaterbodyName', 'Years', 'SampleDate', 'Label'], errors='ignore')

        # Create a copy for processing
        df = data.copy()

        # Outlier Removal using IQR
        for col in df.select_dtypes(include=np.number):
            percentile25 = df[col].quantile(0.25)
            percentile75 = df[col].quantile(0.75)
            iqr = percentile75 - percentile25
            lower_limit = percentile25 - 1.5 * iqr
            upper_limit = percentile75 + 1.5 * iqr
            df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]

        # Normalize data using MinMaxScaler
        scaler = MinMaxScaler()
        features_to_scale = df.columns[df.columns != 'WQI Value']
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

        # Predict WQI values
        new_X = df.drop(columns=['WQI Value'], errors='ignore')
        predictions = loaded_model.predict(new_X)

        # Add predictions to DataFrame
        df['Predicted WQI'] = predictions

        # Display results
        st.write(f"### {selected_model} - Original vs Predicted WQI Values")
        st.dataframe(df[['WQI Value', 'Predicted WQI']])

        # Option to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, "WQI_Comparison.csv", "text/csv")

        st.success(f"Predictions generated successfully using {selected_model}!")

    except Exception as e:
        st.error(f"Error loading dataset or model: {e}")
