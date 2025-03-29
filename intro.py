import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="WQI Prediction App",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0078ff;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0066cc;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Water Quality Index (WQI) Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# Functions
@st.cache_data
def load_data_from_local(file_path):
    """
    Load data from a local file path
    """
    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            return None, f"Error: File not found at {file_path}"
            
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            data = pd.read_excel(file_path)
        else:
            return None, "Error: File must be CSV or Excel format"
            
        return data, f"Successfully loaded {data.shape[0]} rows from {file_path}"
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

@st.cache_data
def preprocess_data(data):
    """
    Apply the same preprocessing steps as during model training
    """
    try:
        # Step 1: Drop unnecessary columns
        data = data.drop(columns=['WaterbodyName', 'Years', 'SampleDate', 'Label'], errors='ignore')
        
        # Step 2: Create a copy for outlier removal
        df = data.copy()
        
        # Step 3: Remove outliers using IQR method
        for col in data.select_dtypes(include=np.number):
            if col == "WQI Value":  # Skip 'WQI Value'
                continue

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
        
        return df, f"Data preprocessing completed."
    except Exception as e:
        return None, f"Error in preprocessing: {str(e)}"

@st.cache_resource
def load_models():
    """
    Load all saved models from the models directory
    """
    # Change path to models folder
    models_dir = "models"
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        return {}, ["❌ Models directory not found. Please create a 'models' folder with model files."]
    
    # List model files from the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
    models = {}
    messages = []
    
    for model_file in model_files:
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ')
        try:
            # Load model from the models directory
            model_path = os.path.join(models_dir, model_file)
            model = joblib.load(model_path)
            models[model_name] = model
            messages.append(f"✅ Loaded model: {model_name}")
        except Exception as e:
            messages.append(f"❌ Error loading {model_file}: {e}")
    
    return models, messages

def predict_wqi(models, features_df, selected_models=None):
    """
    Predict WQI using all or selected models for new data
    """
    predictions = {}
    
    # Filter models if specific ones are selected
    if selected_models and len(selected_models) > 0:
        models_to_use = {name: model for name, model in models.items() if name in selected_models}
    else:
        models_to_use = models
    
    for name, model in models_to_use.items():
        try:
            pred = model.predict(features_df)
            predictions[name] = pred
        except Exception as e:
            st.error(f"Error predicting with {name}: {e}")
    
    return predictions

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# Sidebar
st.sidebar.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzhkdmpvcWswcHVwMnlrM25pYnIxaDZsMmpiYmxrYXBwbGk1M3BwNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Jt5kX3vh7JKeVGpidQ/giphy.gif", width=100)
st.sidebar.title("WQI Prediction Tools")
st.sidebar.markdown("---")

# Define default dataset path
DEFAULT_DATASET_PATH = "Dataset.csv"

# Auto-load data when app starts
@st.cache_data
def auto_load_data():
    data, message = load_data_from_local(DEFAULT_DATASET_PATH)
    return data, message

# Load data automatically
data, message = auto_load_data()

if data is not None:
    st.session_state.data = data
    st.sidebar.success(f"Dataset Load Sucessfully")
else:
    st.sidebar.error(message)

# Load models
models, model_messages = load_models()

if not models:
    st.error("No models found. Make sure model files (.pkl) are in the 'models' directory.")
    st.stop()

# Show model loading status
with st.sidebar.expander("Model Loading Status", expanded=False):
    for msg in model_messages:
        st.write(msg)

# Check if data is loaded
if "data" not in st.session_state or st.session_state.data is None:
    st.error(f"Could not load the dataset from {DEFAULT_DATASET_PATH}. Please make sure the file exists and is accessible.")
    st.stop()

# Main content - Only the Predict New WQI Values functionality
st.markdown("<h2 class='sub-header'>Predict WQI Values</h2>", unsafe_allow_html=True)

new_data = st.session_state.data

# Display raw data sample
with st.expander("Preview Raw Data", expanded=False):
    st.dataframe(new_data.head(), use_container_width=True)

# Check if WQI Value column exists
has_wqi = 'WQI Value' in new_data.columns

# Preprocess data
with st.spinner("Preprocessing data..."):
    processed_data, preprocess_msg = preprocess_data(new_data)
    
if processed_data is None:
    st.error(preprocess_msg)
    st.stop()
else:
    st.info(preprocess_msg)

# Display preprocessed data sample
with st.expander("Preview Preprocessed Data", expanded=False):
    st.dataframe(processed_data.head(), use_container_width=True)

# Prepare features
if has_wqi:
    X_new = processed_data.drop(columns=['WQI Value'])
    y_new = processed_data['WQI Value']
else:
    X_new = processed_data

# Select models to use
st.write("### Select models for prediction")
selected_models = []

# Use checkboxes for each model
for model_name in models.keys():
    if st.checkbox(model_name, value=True):  # Default checked
        selected_models.append(model_name)

# Make predictions button
if st.button("Generate Predictions", type="primary"):
    if not selected_models:
        st.warning("Please select at least one model for prediction")
    else:
        with st.spinner("Generating predictions..."):
            # Make predictions
            predictions = predict_wqi(models, X_new, selected_models)
            
            # Create results DataFrame
            results = pd.DataFrame()
            
            if has_wqi:
                results['Original WQI'] = y_new.reset_index(drop=True)
            
            for name, preds in predictions.items():
                results[f'{name} Prediction'] = preds
                
                if has_wqi:
                    results[f'{name} Error'] = abs(results['Original WQI'] - preds)
            
        # Display predictions table
        st.markdown("<h3 class='sub-header'>Predictions</h3>", unsafe_allow_html=True)
        st.dataframe(results.head(20), use_container_width=True)
        
        # If we have original WQI values, display metrics
        if has_wqi:
            metrics = []
            for name, preds in predictions.items():
                mse = mean_squared_error(y_new, preds)
                r2 = r2_score(y_new, preds)
                rmse = np.sqrt(mse)
                metrics.append({
                    'Model': name,
                    'MSE': mse,
                    'R²': r2,
                    'RMSE': rmse
                })
            
            metrics_df = pd.DataFrame(metrics).sort_values('R²', ascending=False)
            best_model = metrics_df.iloc[0]['Model']
            
            # Display metrics
            st.markdown("<h3 class='sub-header'>Performance Metrics</h3>", unsafe_allow_html=True)
            st.dataframe(metrics_df.style.highlight_max(subset=['R²']).highlight_min(subset=['MSE', 'RMSE']), 
                        use_container_width=True)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot for best model
            axes[0].scatter(results['Original WQI'], results[f'{best_model} Prediction'], alpha=0.5, color='blue')
            min_val = min(results['Original WQI'].min(), results[f'{best_model} Prediction'].min())
            max_val = max(results['Original WQI'].max(), results[f'{best_model} Prediction'].max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[0].set_title(f'Best Model ({best_model}): Predicted vs Actual')
            axes[0].set_xlabel('Actual WQI')
            axes[0].set_ylabel('Predicted WQI')
            
            # Error distribution
            sns.histplot(results[f'{best_model} Error'], kde=True, ax=axes[1], color='green')
            axes[1].set_title(f'Error Distribution for {best_model}')
            axes[1].set_xlabel('Absolute Error')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Download predictions
        st.markdown("### Download Results")
        st.markdown(get_download_link(results, "wqi_predictions.csv", 
                                    "Download Predictions CSV"), unsafe_allow_html=True)
        
        # Download joined data with predictions
        full_results = pd.concat([new_data.reset_index(drop=True), 
                                results.drop('Original WQI', errors='ignore')], axis=1)
        st.markdown(get_download_link(full_results, "full_predictions_with_data.csv", 
                                    "Download Full Results (Original Data + Predictions)"), unsafe_allow_html=True)

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("💧 WQI Prediction App • Made from Danish")
