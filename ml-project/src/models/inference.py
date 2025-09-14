import torch
import numpy as np
from .trainer import TextCNN, pad_sequences, text_to_sequence
from src.data.processor import clean_text
from dataclasses import dataclass
from typing import List
import sys
import src.models.trainer

@dataclass
class InferenceConfig:
    model_path: str = "models/sentiment_cnn.pth"

class SentimentPredictor:
    def __init__(self, config: InferenceConfig):
        # Patch sys.modules so torch.load can find the correct module
        sys.modules['models.trainer'] = src.models.trainer

        checkpoint = torch.load(config.model_path, weights_only=False)
        
        self.vocab_to_int = checkpoint['vocab_to_int']
        self.label_classes = checkpoint['label_encoder_classes']
        self.max_seq_len = checkpoint['max_seq_len']
        train_config = checkpoint['config']
        
        vocab_size = len(self.vocab_to_int) + 1
        num_classes = len(self.label_classes)
        
        self.model = TextCNN(
            vocab_size, 
            train_config.embedding_dim, 
            num_classes, 
            train_config.num_filters, 
            train_config.filter_sizes, 
            train_config.dropout_rate
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, text: str) -> str:
        """Predicts sentiment for a single piece of text."""
        with torch.no_grad():
            cleaned_text = clean_text(text)
            sequence = [text_to_sequence(cleaned_text, self.vocab_to_int)]
            padded_seq = pad_sequences(sequence, self.max_seq_len)
            inputs = torch.from_numpy(padded_seq)
            
            output = self.model(inputs)
            _, predicted_idx = torch.max(output, 1)
            
            return self.label_classes[predicted_idx.item()]

def add_sentiment_prediction(df, predictor: SentimentPredictor):
    """Adds sentiment predictions to a DataFrame."""
    print("-> Running sentiment inference...")
    # Ensure the input column for prediction is 'cleaned_text'
    if 'cleaned_text' not in df.columns and 'review' in df.columns:
        df['cleaned_text'] = df['review'].apply(clean_text)
        
    df['predicted_sentiment'] = df['cleaned_text'].apply(predictor.predict)
    print("✅ Inference complete.")
    return df