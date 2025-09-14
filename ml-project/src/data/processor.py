
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    max_sequence_length: int = 100
    max_vocab_size: int = 10000
    test_size: float = 0.2
    random_state: int = 42
    embedding_dim: int = 194
    
    def __str__(self) -> str:
        return (f"DataConfig(max_sequence_length={self.max_sequence_length}, "
                f"max_vocab_size={self.max_vocab_size}, test_size={self.test_size}, "
                f"random_state={self.random_state}, embedding_dim={self.embedding_dim})")


class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert text to string to handle non-string inputs
        text = str(self.texts[idx]) if not isinstance(self.texts[idx], str) else self.texts[idx]
        
        # Tokenize text (convert to list of word indices)
        tokens = self.tokenize(text)
        
        # Pad or truncate sequence to max_len
        if len(tokens) < self.max_len:
            tokens = tokens + [self.vocab["<pad>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
            
        return torch.tensor(tokens), self.labels[idx]

    def tokenize(self, text):
        # Make sure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        # Split text and convert words to indices
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in text.split()]


class TextProcessor:
    """Class for text data preprocessing."""
    
    def __init__(self, config: DataConfig = None):
        """Initialize with optional config."""
        self.config = config or DataConfig()
        self.vocab = None
        self.stopwords = set(stopwords.words('english'))
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_comments(self, df: pd.DataFrame, comment_cols: List[str]) -> pd.DataFrame:
        """Combine multiple comment columns into a single text field."""
        df = df.copy()
        
        # Create a new column with combined comments
        df['combined_text'] = df[comment_cols].fillna('').apply(lambda x: ' '.join(x), axis=1)
        df['cleaned_text'] = df['combined_text'].apply(self._clean_text)
        
        return df
    
    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from text data.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary mapping words to indices
        """
        # Initialize vocabulary with special tokens
        vocab = {"<pad>": 0, "<unk>": 1}
        word_counts = {}
        
        # Count words in all texts
        for text in texts:
            words = str(text).split()
            for word in words:
                if word not in self.stopwords:  # Skip stopwords
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add top words to vocabulary
        idx = 2  # Start after special tokens
        for word, _ in sorted_words[:self.config.max_vocab_size - 2]:  # -2 for special tokens
            vocab[word] = idx
            idx += 1
        
        self.vocab = vocab
        return vocab
    
    def create_datasets(
        self, 
        train_texts: List[str], 
        train_labels: List[int], 
        val_texts: Optional[List[str]] = None, 
        val_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Create PyTorch datasets for training and validation.
        
        Args:
            train_texts: List of training text strings
            train_labels: List of training labels
            val_texts: List of validation text strings (optional)
            val_labels: List of validation labels (optional)
            
        Returns:
            Dictionary containing train and validation datasets
        """
        # Build vocabulary from training texts
        self.build_vocab(train_texts)
        
        # Create training dataset
        train_dataset = SentimentDataset(
            train_texts, 
            train_labels, 
            self.vocab, 
            max_len=self.config.max_sequence_length
        )
        
        # Create validation dataset if provided
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = SentimentDataset(
                val_texts, 
                val_labels, 
                self.vocab, 
                max_len=self.config.max_sequence_length
            )
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'vocab': self.vocab,
            'vocab_size': len(self.vocab)
        }
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to a file."""
        import pickle
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
            
        with open(path, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.vocab = pickle.load(f)
