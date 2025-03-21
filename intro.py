import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
loaded_model = joblib.load("xgb_model.pkl")

# Upload CSV file
st.title("Water Quality Prediction with XGBoost")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)

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
    st.write("### Original vs Predicted WQI Values")
    st.dataframe(df[['WQI Value', 'Predicted WQI']])

    # Option to download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "WQI_Comparison.csv", "text/csv")

    st.success("Predictions generated successfully!")
