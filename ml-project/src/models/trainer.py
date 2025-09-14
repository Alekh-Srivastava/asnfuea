import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

# Configuration for training
@dataclass
class TrainConfig:
    embedding_dim: int = 100
    num_filters: int = 128
    filter_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    dropout_rate: float = 0.5
    learning_rate: float = 0.001
    epochs: int = 5
    batch_size: int = 50 # Increased batch size for larger dataset

class TextCNN(nn.Module):
    """A Convolutional Neural Network for text classification."""
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [Batch Size, Sequence Length, Embed Dim]
        x = x.unsqueeze(1)  # [Batch Size, 1, Sequence Length, Embed Dim]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

def build_vocab(texts: List[str]) -> Dict[str, int]:
    """Builds a vocabulary from a list of texts."""
    words = [word for text in texts for word in text.split()]
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: i + 1 for i, word in enumerate(vocab)} # +1 for padding token 0
    return vocab_to_int

def text_to_sequence(text: str, vocab_to_int: Dict[str, int]) -> List[int]:
    """Converts a single text to a sequence of integers."""
    return [vocab_to_int.get(word, 0) for word in text.split()]

def pad_sequences(sequences: List[List[int]], max_len: int) -> np.ndarray:
    """Pads sequences to the same length."""
    padded = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        padded[i, -len(seq):] = seq[:max_len]
    return padded

def train_sentiment_model(df: pd.DataFrame, model_save_path: str, config: TrainConfig = TrainConfig()):
    """Trains the CNN model and saves it."""
    print("-> Starting sentiment model training...")

    texts = df['cleaned_text'].tolist()
    labels = df['sentiment'].tolist()

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    vocab_to_int = build_vocab(texts)
    sequences = [text_to_sequence(text, vocab_to_int) for text in texts]
    
    # Cap sequence length for performance
    max_seq_len = 200 
    padded_seqs = pad_sequences(sequences, max_seq_len)

    X_train, X_test, y_train, y_test = train_test_split(padded_seqs, encoded_labels, test_size=0.2, random_state=42)
    
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)

    vocab_size = len(vocab_to_int) + 1 # +1 for the 0 padding
    num_classes = len(label_encoder.classes_)
    model = TextCNN(vocab_size, config.embedding_dim, num_classes, config.num_filters, config.filter_sizes, config.dropout_rate)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {total_loss/len(train_loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_to_int': vocab_to_int,
        'label_encoder_classes': label_encoder.classes_,
        'max_seq_len': max_seq_len,
        'config': config # Save config for inference
    }, model_save_path)
    
    print(f"✅ Model trained and saved to {model_save_path}")
    return model_save_path