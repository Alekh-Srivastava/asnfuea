from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
import sys
import uvicorn

# Add the project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Define the absolute path to models
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

from src.models.inference import SentimentPredictor, InferenceConfig
from src.data.processor import TextProcessor

app = FastAPI(title="Sentiment Analysis API", 
              description="API for sentiment analysis of text data using GRU",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]
    model_name: Optional[str] = "covid_sentiment_gru"

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    
# Global variables
model_cache = {}  # Cache for loaded models

def get_predictor(model_name: str = "covid_sentiment_gru") -> SentimentPredictor:
    """Get or create a SentimentPredictor instance."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for model: {model_name}")
    
    if model_name in model_cache:
        return model_cache[model_name]
    
    # Set up model paths with absolute paths
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    vocab_path = os.path.join(MODELS_DIR, "vocab.pkl")
    
    print(f"Model path: {model_path}")
    print(f"Vocab path: {vocab_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Vocab exists: {os.path.exists(vocab_path)}")
    
    # Check if model and vocabulary exist
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found at {model_path}")
    
    if not os.path.exists(vocab_path):
        raise HTTPException(status_code=404, detail=f"Vocabulary file not found at {vocab_path}")
    
    # Create and load predictor
    config = InferenceConfig(model_path=model_path, vocab_path=vocab_path)
    predictor = SentimentPredictor(config=config)
    predictor.load_model()
    
    # Cache predictor
    model_cache[model_name] = predictor
    
    return predictor

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.get("/models")
def list_models():
    """List available models."""
    # List .pt files in the models directory
    if not os.path.exists(MODELS_DIR):
        return {"models": []}
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    model_names = [os.path.splitext(f)[0] for f in model_files]
    
    return {"models": model_names}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: TextRequest, model_name: str = "covid_sentiment_gru"):
    """
    Predict sentiment for a single text.
    
    Args:
        request: TextRequest with text field
        model_name: Name of the model to use (default: covid_sentiment_gru)
        
    Returns:
        PredictionResponse with sentiment prediction
    """
    try:
        predictor = get_predictor(model_name)
        
        # Make prediction
        result = predictor.predict([request.text])
        
        return PredictionResponse(
            text=request.text,
            sentiment=result['labels'][0],
            confidence=result['confidence'][0]
        )
    except Exception as e:
        print(f"Error in predict_sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchTextRequest):
    """
    Predict sentiment for a batch of texts.
    
    Args:
        request: BatchTextRequest with texts list and optional model_name
        
    Returns:
        BatchPredictionResponse with sentiment predictions
    """
    try:
        predictor = get_predictor(request.model_name)
        
        # Make predictions
        result = predictor.predict(request.texts)
        
        # Format response
        predictions = [
            PredictionResponse(
                text=text,
                sentiment=label,
                confidence=confidence
            )
            for text, label, confidence in zip(
                request.texts, 
                result['labels'], 
                result['confidence']
            )
        ]
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        print(f"Error in predict_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), model_name: str = "covid_sentiment_gru"):
    """
    Upload a CSV file for batch prediction.
    
    Args:
        file: CSV file with text data
        model_name: Name of the model to use (default: covid_sentiment_gru)
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
        
        # Validate that there's a text column
        text_column = None
        for col in ['text', 'comment', 'combined_text', 'cleaned_text']:
            if col in df.columns:
                text_column = col
                break
                
        if text_column is None:
            # Look for Comment columns
            comment_cols = [col for col in df.columns if col.startswith('Comment')]
            if comment_cols:
                # Combine comments
                processor = TextProcessor()
                df = processor.combine_comments(df, comment_cols)
                text_column = 'cleaned_text'
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No text column found in CSV. Please include a column named 'text', 'comment', or 'Comment'"
                )
        
        # Get predictor
        predictor = get_predictor(model_name)
        
        # Make predictions
        result_df = predictor.predict_dataframe(df, text_column)
        
        # Convert to JSON response
        result_json = result_df.to_json(orient='records')
        result_data = json.loads(result_json)
        
        return {
            "filename": file.filename,
            "rows_processed": len(result_df),
            "predictions": result_data
        }
    except Exception as e:
        print(f"Error in upload_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-json")
async def upload_json(file: UploadFile = File(...), model_name: str = "covid_sentiment_gru"):
    """
    Upload a JSON file for batch prediction.
    
    Args:
        file: JSON file with text data
        model_name: Name of the model to use (default: covid_sentiment_gru)
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Read JSON
        content = await file.read()
        data = json.loads(content)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        # Validate that there's a text column
        text_column = None
        for col in ['text', 'comment', 'combined_text', 'cleaned_text']:
            if col in df.columns:
                text_column = col
                break
                
        if text_column is None:
            # Look for Comment columns
            comment_cols = [col for col in df.columns if col.startswith('Comment')]
            if comment_cols:
                # Combine comments
                processor = TextProcessor()
                df = processor.combine_comments(df, comment_cols)
                text_column = 'cleaned_text'
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No text column found in JSON. Please include a field named 'text', 'comment', or 'Comment'"
                )
        
        # Get predictor
        predictor = get_predictor(model_name)
        
        # Make predictions
        result_df = predictor.predict_dataframe(df, text_column)
        
        # Convert to JSON response
        result_json = result_df.to_json(orient='records')
        result_data = json.loads(result_json)
        
        return {
            "filename": file.filename,
            "rows_processed": len(result_df),
            "predictions": result_data
        }
    except Exception as e:
        print(f"Error in upload_json: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting server with models directory: {MODELS_DIR}")
    print(f"Available models: {[f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]}")
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)