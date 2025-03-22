import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Streamlit App Title
st.title("Water Quality Prediction - Model Comparison")

# Dictionary of Models (Ensure these files exist)
models = {
    "XGBoost": "model/xgb_model.pkl",
    "Decision Tree": "model/dt_model.pkl",
    "AdaBoost": "model/ab_model.pkl",
    "Multi-Layer Perceptron": "model/mlp_model.pkl",
    "Support Vector Machine": "model/svm_model.pkl",
    "Gradient Boosting": "model/gb_model.pkl"
}

# GitHub raw dataset URL (Replace with your actual dataset link)
dataset_path = "Dataset.csv"

# Button to load dataset and compare models
if st.button("Load Dataset and Compare Models"):
    try:
        # Load dataset
        data = pd.read_csv(dataset_path)

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

        # Prepare input features
        new_X = df.drop(columns=['WQI Value'], errors='ignore')

        # Store results
        model_results = []
        predictions_df = df[['WQI Value']].copy()

        # Loop through each model, predict, and evaluate
        for model_name, model_file in models.items():
            try:
                # Load Model
                loaded_model = joblib.load(model_file)

                # Predict WQI values
                predictions = loaded_model.predict(new_X)
                predictions_df[model_name] = predictions

                # Calculate performance metrics
                mse = mean_squared_error(df['WQI Value'], predictions)
                r2 = r2_score(df['WQI Value'], predictions)
                mae = mean_absolute_error(df['WQI Value'], predictions)

                # Store results
                model_results.append({
                    "Model": model_name,
                    "MSE": mse,
                    "R² Score": r2,
                    "MAE": mae
                })

            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")

        # Convert results to DataFrame
        results_df = pd.DataFrame(model_results)

        # Display Results Table
        st.write("### Model Performance Comparison")
        st.dataframe(results_df)

        # Display predictions
        st.write("### Original vs Predicted WQI Values")
        st.dataframe(predictions_df)

        # Option to download results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Model Comparison CSV", csv, "Model_Comparison.csv", "text/csv")

        st.success("Model comparison completed successfully!")

    except Exception as e:
        st.error(f"Error loading dataset or models: {e}")
