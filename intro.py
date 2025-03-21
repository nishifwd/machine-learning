import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the trained model
model = joblib.load("xgb_model.pkl")

# Streamlit App Title
st.title("Water Quality Prediction with XGBoost")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Ensure 'WQI Value' exists before dropping
    if 'WQI Value' in df.columns:
        # Drop target column to use as features
        new_X = df.drop(columns=['WQI Value', 'WaterbodyName', 'Years', 'SampleDate', 'Label'])

        # Make predictions
        predictions = model.predict(new_X)

        # Add predictions to DataFrame
        df['Predicted WQI'] = predictions

        # Display Comparison Table
        st.write("### Original vs Predicted WQI")
        st.write(df[['WQI Value', 'Predicted WQI']])

        # Save file for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="WQI_Comparison.csv",
            mime="text/csv",
        )

        # Calculate Performance Metrics
        mse = mean_squared_error(df['WQI Value'], df['Predicted WQI'])
        r2 = r2_score(df['WQI Value'], df['Predicted WQI'])
        mae = mean_absolute_error(df['WQI Value'], df['Predicted WQI'])

        # Display Metrics
        st.write("### Model Performance Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R-squared (R²):** {r2:.4f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")

        # Visualization: Actual vs Predicted WQI
        st.write("### Actual vs Predicted WQI Graph")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df['WQI Value'].values, label="Actual WQI", marker='o', linestyle='dashed')
        ax.plot(df['Predicted WQI'].values, label="Predicted WQI", marker='x', linestyle='solid')
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("WQI Value")
        ax.set_title("Actual vs Predicted WQI")
        ax.legend()
        st.pyplot(fig)

    else:
        st.error("The uploaded dataset must contain the column 'WQI Value'.")
