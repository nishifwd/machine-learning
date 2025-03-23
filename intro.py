import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import os
import requests
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def load_data_from_github(github_url):
    """
    Load dataset from a raw GitHub URL
    """
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Check if the request was successful
        data = pd.read_csv(pd.compat.StringIO(response.text))
        st.write("Dataset loaded successfully from GitHub")
        return data
    except requests.exceptions.RequestException as e:
        st.write(f"Error loading dataset from GitHub: {e}")
        return None

def preprocess_data(data):
    """
    Apply the same preprocessing steps as during model training
    """
    # Step 1: Drop unnecessary columns
    data = data.drop(columns=['WaterbodyName', 'Years', 'SampleDate', 'Label'], errors='ignore')
    
    # Step 2: Create a copy for outlier removal
    df = data.copy()
    
    # Step 3: Remove outliers using IQR method
    for col in data.select_dtypes(include=np.number):
        # Calculate the 25th and 75th percentiles (Q1 and Q3)
        percentile25 = df[col].quantile(0.25)
        percentile75 = df[col].quantile(0.75)

        # Calculate the IQR
        iqr = percentile75 - percentile25

        # Set the lower and upper bounds for outliers
        lower_limit = percentile25 - 1.5 * iqr
        upper_limit = percentile75 + 1.5 * iqr

        # Remove outliers by filtering the DataFrame
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    
    # Step 4: Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    features_to_scale = df.columns[df.columns != 'WQI Value']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    st.write(f"Data preprocessing completed. Rows after outlier removal: {df.shape[0]}")
    
    return df

def load_models(model_directory='models'):
    """
    Load all saved models from a specified directory
    """
    model_files = [f for f in os.listdir(model_directory) if f.endswith('_model.pkl')]
    models = {}
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
        try:
            model_path = os.path.join(model_directory, model_file)
            model = joblib.load(model_path)
            models[model_name] = model
            st.write(f"Loaded model: {model_name}")
        except Exception as e:
            st.write(f"Error loading {model_file}: {e}")
    
    return models

def test_models(models, X_test, y_test):
    """
    Test all models on the test dataset and return performance metrics
    """
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        npe = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rmse = np.sqrt(mse)
        
        results[name] = {
            'predictions': y_pred,
            'mse': mse,
            'r2': r2,
            'npe': npe,
            'rmse': rmse
        }
        
    return results

def predict_wqi(models, features_df):
    """
    Predict WQI using all models for new data
    """
    predictions = {}
    
    for name, model in models.items():
        try:
            pred = model.predict(features_df)
            predictions[name] = pred
        except Exception as e:
            st.write(f"Error predicting with {name}: {e}")
    
    return predictions

def compare_predictions(original_wqi, predictions, dataset_name="Test Data"):
    """
    Create a DataFrame comparing original WQI with predicted values from different models
    """
    results_df = pd.DataFrame()
    results_df['Original WQI'] = original_wqi.reset_index(drop=True)
    
    for model_name, preds in predictions.items():
        # Ensure predictions have the same length as original_wqi
        if len(preds) == len(original_wqi):
            results_df[f'{model_name} Prediction'] = preds
            results_df[f'{model_name} Error'] = abs(results_df['Original WQI'] - preds)
        else:
            st.write(f"Warning: Length mismatch for {model_name}. Predictions: {len(preds)}, Original: {len(original_wqi)}")
    
    return results_df

def visualize_results(results, results_df, test_data_name="Test Dataset"):
    """
    Create visualizations to compare model performance
    """
    # 1. Model performance metrics comparison
    metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[model]['mse'] for model in results],
        'R²': [results[model]['r2'] for model in results],
        'RMSE': [results[model]['rmse'] for model in results],
        'NPE (%)': [results[model]['npe'] for model in results]
    })
    
    # Sort by R² (higher is better)
    metrics = metrics.sort_values('R²', ascending=False)
    
    st.write("Model Performance Metrics:")
    st.write(metrics)
    
    # 2. Plot predicted vs actual values for the best model
    best_model = metrics.iloc[0]['Model']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Bar chart of performance metrics
    metrics[['Model', 'R²']].set_index('Model').plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Model R² Comparison')
    axes[0, 0].set_ylabel('R² Score (higher is better)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(metrics['Model'], metrics['RMSE'])
    axes[0, 1].set_title('Model RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE (lower is better)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Scatter plot of best model predictions
    axes[1, 0].scatter(results_df['Original WQI'], results_df[f'{best_model} Prediction'], alpha=0.6)
    axes[1, 0].plot([results_df['Original WQI'].min(), results_df['Original WQI'].max()], 
                    [results_df['Original WQI'].min(), results_df['Original WQI'].max()], 'r--')
    axes[1, 0].set_title(f'Best Model ({best_model}): Predicted vs Actual WQI')
    axes[1, 0].set_xlabel('Actual WQI')
    axes[1, 0].set_ylabel('Predicted WQI')
    
    # Plot 4: Error distribution for best model
    sns.histplot(results_df[f'{best_model} Error'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title(f'Error Distribution for {best_model}')
    axes[1, 1].set_xlabel('Absolute Error')
    
    st.pyplot(fig)
    
    # 3. Detailed comparison of predictions for a few samples
    sample_comparison = results_df.head(10)  # Show first 10 rows
    st.write(f"\nSample Comparison ({test_data_name}):")
    st.write(sample_comparison)

    return metrics, best_model

def main():
    st.title("Water Quality Index (WQI) Prediction and Model Evaluation")
    
    # GitHub URL for the dataset (change this to your raw GitHub dataset URL)
    github_url = 'https://github.com/nishifwd/machine-learning/blob/main/Dataset.csv'
    
    # Load dataset from GitHub
    data = load_data_from_github(github_url)
    
    if data is not None:
        # Apply the same preprocessing steps as in training
        df = preprocess_data(data)
        
        # Define X (features) and y (target) after preprocessing
        X = df.drop(columns=['WQI Value'])  # Features (input variables)
        y = df['WQI Value']  # Target (output variable)
        
        # Load all saved models from the 'model' directory
        models = load_models(model_directory='model')
        
        if not models:
            st.write("No models found. Make sure model files are in the 'model' directory.")
            return
        
        # Test on a split from the preprocessed data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write("\nTesting models on the test dataset...")
        test_results = test_models(models, X_test, y_test)
        test_predictions = {name: test_results[name]['predictions'] for name in test_results}
        test_comparison_df = compare_predictions(y_test, test_predictions, "Test Data")
        
        _, best_model = visualize_results(test_results, test_comparison_df)
        
        # Generate predictions for the entire dataset
        st.write("\nGenerating predictions for the entire dataset...")
        full_predictions = predict_wqi(models, X)
        full_comparison_df = compare_predictions(y, full_predictions, "Full Dataset")
        
        # Save comparison results to CSV files
        test_comparison_df.to_csv('test_data_predictions.csv', index=False)
        full_comparison_df.to_csv('full_dataset_predictions.csv', index=False)
        
        st.write(f"\nBest performing model: {best_model}")
        st.write("\nResults have been saved to:")
        st.write("- test_data_predictions.csv (Test data predictions)")
        st.write("- full_dataset_predictions.csv (Full dataset predictions)")

if __name__ == "__main__":
    main()
