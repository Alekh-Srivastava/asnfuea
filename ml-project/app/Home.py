# pylint: disable=invalid-name

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import io
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page config
st.set_page_config(
    page_title="GRU Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://0.0.0.0:8000"

def get_available_models():
    """Get list of available models from the API."""
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            return response.json()["models"]
        return ["covid_sentiment_gru"]  # Default if API call fails
    except:
        return ["covid_sentiment_gru"]  # Default if API is not running

def predict_single_text(text, model_name):
    """Make a prediction for a single text."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            params={"model_name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def predict_batch_texts(texts, model_name):
    """Make predictions for a batch of texts."""
    try:
        response = requests.post(
            f"{API_URL}/predict-batch",
            json={"texts": texts, "model_name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def upload_csv(file, model_name):
    """Upload a CSV file for prediction."""
    try:
        files = {"file": file}
        response = requests.post(
            f"{API_URL}/upload-csv",
            files=files,
            params={"model_name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def upload_json(file, model_name):
    """Upload a JSON file for prediction."""
    try:
        files = {"file": file}
        response = requests.post(
            f"{API_URL}/upload-json",
            files=files,
            params={"model_name": model_name}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def plot_sentiment_distribution(df):
    """Plot sentiment distribution."""
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        sns.barplot(x='Sentiment', y='Count', data=sentiment_counts, palette=colors, ax=ax)
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        
        # Add count labels on top of bars
        for i, p in enumerate(ax.patches):
            ax.annotate(
                f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
        
        st.pyplot(fig)

def plot_confidence_distribution(df):
    """Plot confidence score distribution."""
    if 'sentiment_confidence' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['sentiment_confidence'], bins=20, kde=True, ax=ax)
        ax.set_title('Confidence Score Distribution')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        st.pyplot(fig)

def main():
    """Main function for the Streamlit app."""
    st.title("GRU Sentiment Analysis App")
    
    st.markdown("""
    This app analyzes the sentiment of text using a GRU (Gated Recurrent Unit) model trained on social media data.
    Upload your data or enter text directly to get sentiment predictions.
    """)
    
    # Sidebar with model selection
    st.sidebar.title("Model Selection")
    available_models = get_available_models()
    
    if available_models:
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["all"] + available_models if len(available_models) > 1 else available_models,
            help="Select a model for sentiment analysis. 'all' will use all available models."
        )
    else:
        model_name = "covid_sentiment_gru"
        st.sidebar.warning("No models available. Using default model.")
    
    # Display model information
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("""
    **GRU Model Parameters:**
    - Embedding Dim: 194
    - Hidden Dim: 243
    - GRU Layers: 3
    - Dropout Rate: 0.104
    - Batch Size: 64
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Upload", "Data Exploration"])
    
    # Single text analysis
    with tab1:
        st.header("Analyze Single Text")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            "The government's response to COVID-19 has been impressive. They've acted quickly and responsibly.",
            height=150
        )
        
        if st.button("Analyze Sentiment"):
            if text_input:
                with st.spinner("Analyzing sentiment..."):
                    result = predict_single_text(text_input, model_name)
                
                if result:
                    # Display result
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]
                    
                    # Set color based on sentiment
                    if sentiment == "Positive":
                        color = "green"
                    elif sentiment == "Negative":
                        color = "red"
                    else:
                        color = "blue"
                    
                    st.markdown(f"### Sentiment: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
                    st.markdown(f"### Confidence: {confidence:.2%}")
                    
                    # Add confidence meter
                    st.progress(confidence)
            else:
                st.warning("Please enter some text to analyze.")
    
    # Batch upload
    with tab2:
        st.header("Batch Analysis")
        
        upload_type = st.radio("Select file type:", ["CSV", "JSON"])
        
        uploaded_file = st.file_uploader(
            f"Upload a {upload_type} file",
            type=["csv"] if upload_type == "CSV" else ["json"]
        )
        
        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded successfully.")
            
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    if upload_type == "CSV":
                        result = upload_csv(uploaded_file, model_name)
                    else:
                        result = upload_json(uploaded_file, model_name)
                
                if result:
                    st.success(f"Processed {result['rows_processed']} rows of data.")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(result['predictions'])
                    
                    # Display results
                    st.subheader("Results")
                    st.dataframe(df)
                    
                    # Option to download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
                    
                    # Visualizations
                    st.subheader("Visualizations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        plot_sentiment_distribution(df)
                    
                    with col2:
                        plot_confidence_distribution(df)
    
    # Data exploration
    with tab3:
        st.header("Data Exploration")
        
        st.info("Upload a file to explore the data.")
        
        explore_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            key="explore_upload"
        )
        
        if explore_file is not None:
            # Load data
            df = pd.read_csv(explore_file)
            
            st.subheader("Data Overview")
            st.dataframe(df.head())
            
            # Display basic statistics
            st.subheader("Data Statistics")
            st.write(df.describe())
            
            # Display column information
            st.subheader("Column Information")
            
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            
            st.dataframe(col_info)
            
            # Column selection for visualization
            st.subheader("Visualize Column")
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                selected_num_col = st.selectbox("Select numeric column:", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {selected_num_col}')
                st.pyplot(fig)
            
            if categorical_cols:
                selected_cat_col = st.selectbox("Select categorical column:", categorical_cols)
                
                # Limit to top 10 categories if there are many unique values
                value_counts = df[selected_cat_col].value_counts().nlargest(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f'Top 10 values for {selected_cat_col}')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()