from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import json
import torch
import numpy as np
import pandas as pd
import pickle

from src.models.trainer import GRUModel, TrainConfig


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    model_path: str = 'models/sentiment_gru.pt'
    vocab_path: str = 'models/vocab.pkl'
    batch_size: int = 64
    max_sequence_length: int = 100
    
    def __str__(self) -> str:
        """String representation of the config."""
        return (f"InferenceConfig(model_path={self.model_path}, "
                f"vocab_path={self.vocab_path}, batch_size={self.batch_size}, "
                f"max_sequence_length={self.max_sequence_length})")


class SentimentPredictor:
    """Class for making sentiment predictions with GRU model."""
    
    def __init__(self, config: InferenceConfig = None):
        """Initialize with optional config."""
        self.config = config or InferenceConfig()
        self.model = None
        self.vocab = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Default mapping
        
    def load_model(self) -> None:
        """Load the trained model and vocabulary."""
        # Load vocabulary
        if not os.path.exists(self.config.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {self.config.vocab_path}")
            
        with open(self.config.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # Load model config
        config_path = os.path.splitext(self.config.model_path)[0] + '_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            train_config = TrainConfig(**config_data['config'])
            vocab_size = config_data.get('vocab_size', len(self.vocab))
            num_classes = config_data.get('num_classes', 3)
            
            # Check if label mapping is in config
            if 'label_map' in config_data:
                self.label_map = config_data['label_map']
        else:
            # Use default config
            train_config = TrainConfig()
            vocab_size = len(self.vocab)
            num_classes = 3
        
        # Initialize model
        self.model = GRUModel(
            vocab_size=vocab_size,
            embedding_dim=train_config.embedding_dim,
            hidden_dim=train_config.hidden_dim,
            output_dim=num_classes,
            n_layers=train_config.n_layers,
            dropout=train_config.dropout_rate
        )
        
        # Load model weights
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model and vocabulary loaded successfully")
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        text = self._clean_text(text)
        words = text.split()
        tokens = [self.vocab.get(word, self.vocab.get("<unk>", 1)) for word in words]
        
        # Pad or truncate to max_sequence_length
        if len(tokens) < self.config.max_sequence_length:
            tokens = tokens + [self.vocab.get("<pad>", 0)] * (self.config.max_sequence_length - len(tokens))
        else:
            tokens = tokens[:self.config.max_sequence_length]
            
        return tokens
    
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make sentiment predictions on new texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.model is None or self.vocab is None:
            self.load_model()
        
        # Tokenize texts
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # Convert to tensor
        input_tensor = torch.tensor(tokenized_texts).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Get predicted classes and confidence scores
        predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
        confidence_scores = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        # Map class IDs to labels
        predicted_labels = [self.label_map.get(str(int(class_id)), f"Class {class_id}") 
                   for class_id in predicted_classes]
        
        return {
            'texts': texts,
            'predictions': predicted_classes.tolist(),
            'labels': predicted_labels,
            'confidence': confidence_scores.tolist(),
            'probabilities': probabilities.cpu().numpy().tolist()
        }
    
    def predict_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Make predictions on a dataframe and add results as new columns.
        
        Args:
            df: DataFrame containing text data
            text_col: Name of the column containing text to analyze
            
        Returns:
            DataFrame with added prediction columns
        """
        texts = df[text_col].tolist()
        results = self.predict(texts)
        
        # Add predictions to the dataframe
        df = df.copy()
        df['sentiment_prediction'] = results['predictions']
        df['sentiment_label'] = results['labels']
        df['sentiment_confidence'] = results['confidence']
        
        return df