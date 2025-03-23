import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import io
import base64
import requests

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
def load_data_from_github(url):
    """
    Load data from a GitHub repository URL
    """
    try:
        # Convert GitHub page URL to raw content URL if needed
        if "github.com" in url and "blob" in url:
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        else:
            raw_url = url
            
        # Fetch data
        data = pd.read_csv(raw_url)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Make sure your URL points to a raw CSV file or valid GitHub CSV file path")
        return None

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
        
        return df, f"Data preprocessing completed. Rows after outlier removal: {df.shape[0]}"
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

def test_models(models, X_test, y_test):
    """
    Test all models on the test dataset and return performance metrics
    """
    results = {}
    
    for name, model in models.items():
        with st.spinner(f"Testing {name} model..."):
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

def compare_predictions(original_wqi, predictions):
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
            st.warning(f"Length mismatch for {model_name}. Predictions: {len(preds)}, Original: {len(original_wqi)}")
    
    return results_df

def visualize_results(results, results_df):
    """
    Create visualizations to compare model performance
    """
    # Model performance metrics comparison
    metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[model]['mse'] for model in results],
        'R²': [results[model]['r2'] for model in results],
        'RMSE': [results[model]['rmse'] for model in results],
        'NPE (%)': [results[model]['npe'] for model in results]
    })
    
    # Sort by R² (higher is better)
    metrics = metrics.sort_values('R²', ascending=False)
    
    # Find best model
    best_model = metrics.iloc[0]['Model']
    
    # Display metrics table
    st.markdown("<h3 class='sub-header'>Model Performance Metrics</h3>", unsafe_allow_html=True)
    st.dataframe(metrics.style.highlight_max(subset=['R²']).highlight_min(subset=['MSE', 'RMSE', 'NPE (%)']), use_container_width=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Bar chart of R² scores
    metrics[['Model', 'R²']].sort_values('R²', ascending=True).plot(
        kind='barh', x='Model', y='R²', ax=axes[0, 0], legend=False, color='skyblue'
    )
    axes[0, 0].set_title('Model R² Comparison (higher is better)')
    axes[0, 0].set_xlabel('R² Score')
    
    # Plot 2: Bar chart of RMSE
    metrics[['Model', 'RMSE']].sort_values('RMSE', ascending=False).plot(
        kind='barh', x='Model', y='RMSE', ax=axes[0, 1], legend=False, color='salmon'
    )
    axes[0, 1].set_title('Model RMSE Comparison (lower is better)')
    axes[0, 1].set_xlabel('RMSE')
    
    # Plot 3: Scatter plot of best model predictions
    axes[1, 0].scatter(results_df['Original WQI'], results_df[f'{best_model} Prediction'], alpha=0.6, color='blue')
    min_val = min(results_df['Original WQI'].min(), results_df[f'{best_model} Prediction'].min())
    max_val = max(results_df['Original WQI'].max(), results_df[f'{best_model} Prediction'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1, 0].set_title(f'Best Model ({best_model}): Predicted vs Actual WQI')
    axes[1, 0].set_xlabel('Actual WQI')
    axes[1, 0].set_ylabel('Predicted WQI')
    
    # Plot 4: Error distribution for best model
    error_col = f'{best_model} Error'
    sns.histplot(results_df[error_col], kde=True, ax=axes[1, 1], color='green')
    axes[1, 1].set_title(f'Error Distribution for {best_model}')
    axes[1, 1].set_xlabel('Absolute Error')
    
    plt.tight_layout()
    
    # Display plot in Streamlit
    st.pyplot(fig)
    
    # Generate and return a download link for the plot
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    
    return metrics, best_model, plot_data

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

# Sidebar
st.sidebar.image("https://www.epa.gov/sites/default/files/2016-03/epa_seal_verysmall_trim.gif", width=100)
st.sidebar.title("WQI Prediction Tools")
st.sidebar.markdown("---")

# Data source input
st.sidebar.subheader("Data Source")
data_source = st.sidebar.radio("Select data source", ["GitHub URL", "Sample Data"])

if data_source == "GitHub URL":
    default_url = "https://raw.githubusercontent.com/nishifwd/machine-learning/main/Dataset.csv"
    data_url = st.sidebar.text_input("Enter GitHub URL", value=default_url)
    
    # Load data button
    if st.sidebar.button("Load Data", type="primary"):
        data = load_data_from_github(data_url)
        if data is not None:
            st.session_state.data = data
            st.sidebar.success(f"Data loaded successfully: {data.shape[0]} rows")
else:  # Sample data
    st.sidebar.info("Using built-in sample data")
    # Create sample data or load from a local file
    # This is just an example, adjust according to your data structure
    sample_data = pd.DataFrame({
        'DO': np.random.uniform(4, 9, 100),
        'pH': np.random.uniform(6.5, 8.5, 100),
        'BOD': np.random.uniform(1, 10, 100),
        'Nitrate': np.random.uniform(0, 20, 100),
        'Fecal Coliform': np.random.uniform(0, 500, 100),
        'Turbidity': np.random.uniform(0, 50, 100),
        'Temperature': np.random.uniform(15, 30, 100),
        'WQI Value': np.random.uniform(20, 100, 100)
    })
    
    if st.sidebar.button("Use Sample Data", type="primary"):
        st.session_state.data = sample_data
        st.sidebar.success("Sample data loaded successfully")

# Load models
models, model_messages = load_models()

if not models:
    st.error("No models found. Make sure model files (.pkl) are in the 'models' directory.")
    st.stop()

# Show model loading status
with st.sidebar.expander("Model Loading Status", expanded=False):
    for msg in model_messages:
        st.write(msg)

# Sidebar navigation
page = st.sidebar.radio("Select Mode", ["Test on Loaded Data", "Predict New WQI Values"])

# Check if data is loaded
if "data" not in st.session_state:
    st.info("Please load data using the sidebar options")
    st.stop()

# Main content based on page selection
if page == "Test on Loaded Data":
    st.markdown("<h2 class='sub-header'>Test Models on Your Data</h2>", unsafe_allow_html=True)
    
    data = st.session_state.data
    st.success(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns")
    
    # Display raw data sample
    with st.expander("Preview Raw Data", expanded=False):
        st.dataframe(data.head(), use_container_width=True)
    
    # Check if WQI Value column exists
    if 'WQI Value' not in data.columns:
        st.error("The dataset must contain a 'WQI Value' column to evaluate model performance.")
        st.stop()
    
    # Preprocess data
    with st.spinner("Preprocessing data..."):
        df, preprocess_msg = preprocess_data(data)
        
    if df is None:
        st.error(preprocess_msg)
        st.stop()
    else:
        st.info(preprocess_msg)
    
    # Display preprocessed data sample
    with st.expander("Preview Preprocessed Data", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
    
    # Define X and y
    X = df.drop(columns=['WQI Value'])
    y = df['WQI Value']
    
    # Test set size slider
    test_size = st.slider("Select test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.info(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets")
    
    # Select models to test
    selected_models = st.multiselect("Select models to test (default: all)", 
                                    options=list(models.keys()),
                                    default=list(models.keys()))
    
    # Run test button
    if st.button("Run Tests", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model to test")
        else:
            filtered_models = {name: model for name, model in models.items() if name in selected_models}
            
            with st.spinner("Testing models..."):
                # Test models
                results = test_models(filtered_models, X_test, y_test)
                
                # Get predictions
                test_predictions = {name: results[name]['predictions'] for name in results}
                
                # Compare predictions
                test_comparison_df = compare_predictions(y_test, test_predictions)
                
            # Display results
            metrics_df, best_model, plot_data = visualize_results(results, test_comparison_df)
            
            # Display sample predictions
            st.markdown("<h3 class='sub-header'>Sample Predictions</h3>", unsafe_allow_html=True)
            st.dataframe(test_comparison_df.head(10), use_container_width=True)
            
            # Best model summary
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Best Performing Model: {best_model}</h3>
                <p>R² Score: {metrics_df.loc[metrics_df['Model'] == best_model, 'R²'].values[0]:.4f}</p>
                <p>RMSE: {metrics_df.loc[metrics_df['Model'] == best_model, 'RMSE'].values[0]:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download options
            st.markdown("### Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(get_download_link(test_comparison_df, "test_predictions.csv", 
                                            "Download Predictions CSV"), unsafe_allow_html=True)
            
            with col2:
                st.markdown(get_download_link(metrics_df, "model_metrics.csv", 
                                            "Download Metrics CSV"), unsafe_allow_html=True)
            
            # Download plot
            st.markdown(f"""
            <a href="data:image/png;base64,{plot_data}" download="model_comparison_plot.png">
            📥 Download Performance Plot
            </a>
            """, unsafe_allow_html=True)

else:  # Predict New WQI Values
    st.markdown("<h2 class='sub-header'>Predict WQI Values for New Data</h2>", unsafe_allow_html=True)
    
    new_data = st.session_state.data
    st.success(f"Dataset loaded with {new_data.shape[0]} rows and {new_data.shape[1]} columns")
    
    # Display raw data sample
    with st.expander("Preview Raw Data", expanded=False):
        st.dataframe(new_data.head(), use_container_width=True)
    
    # Check if WQI Value column exists
    has_wqi = 'WQI Value' in new_data.columns
    
    if has_wqi:
        st.info("The dataset contains a 'WQI Value' column. We'll compare predictions with actual values.")
    else:
        st.info("The dataset does not contain a 'WQI Value' column. We'll only generate predictions.")
    
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
    selected_models = st.multiselect("Select models for prediction (default: all)", 
                                    options=list(models.keys()),
                                    default=list(models.keys()))
    
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
st.sidebar.info("💧 WQI Prediction App • Built with Streamlit")
