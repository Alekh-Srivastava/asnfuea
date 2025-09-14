
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class TrainConfig:
    """Configuration for model training."""
    embedding_dim: int = 194
    hidden_dim: int = 243
    n_layers: int = 3
    dropout_rate: float = 0.104
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.0032
    
    def __str__(self) -> str:
        """String representation of the config."""
        return (f"TrainConfig(embedding_dim={self.embedding_dim}, "
                f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, "
                f"dropout_rate={self.dropout_rate}, batch_size={self.batch_size}, "
                f"epochs={self.epochs}, learning_rate={self.learning_rate})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }


class GRUModel(nn.Module):
    """GRU model for sentiment analysis."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)  # hidden: [n_layers, batch_size, hidden_dim]
        hidden = hidden[-1, :, :]  # Get the last layer's hidden state [batch_size, hidden_dim]
        return self.fc(self.dropout(hidden))  # [batch_size, output_dim]


class SentimentTrainer:
    """Class for training GRU sentiment models."""
    
    def __init__(self, config: TrainConfig = None):
        """Initialize with optional config."""
        self.config = config or TrainConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = None
        self.optimizer = None
        
    def build_model(self, vocab_size: int, num_classes: int) -> GRUModel:
        """
        Build a GRU model for text classification.
        
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            
        Returns:
            GRU model
        """
        model = GRUModel(
            vocab_size=vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=num_classes,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout_rate
        )
        
        model = model.to(self.device)
        self.model = model
        
        # Set up criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        return model
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (text, labels) in enumerate(progress_bar):
            text, labels = text.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(text)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return epoch_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, List[int], List[int]]:
        """
        Evaluate the model.
        
        Args:
            data_loader: DataLoader for evaluation data
            
        Returns:
            Tuple of (average loss, predictions, true labels)
        """
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating")
            for batch_idx, (text, labels) in enumerate(progress_bar):
                text, labels = text.to(self.device), labels.to(self.device)
                
                predictions = self.model(text)
                loss = self.criterion(predictions, labels)
                
                epoch_loss += loss.item()
                
                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return epoch_loss / len(data_loader), all_preds, all_labels
    
    def train(
        self, 
        train_dataset, 
        val_dataset=None, 
        vocab_size: int = None,
        num_classes: int = None,
        model_path: str = 'models/sentiment_gru.pt'
    ) -> Dict[str, Any]:
        """
        Train the GRU model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            model_path: Path to save the model
            
        Returns:
            Dictionary with training history and evaluation metrics
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size
            )
        
        # Determine vocabulary size and number of classes if not provided
        if vocab_size is None:
            vocab_size = max(getattr(train_dataset, 'vocab_size', 0), 
                           getattr(train_dataset, 'vocab', {}).get('vocab_size', 0))
            
            # If still not found, try to determine from data
            if vocab_size == 0:
                # Get the first batch and find the maximum token value
                sample_batch, _ = next(iter(train_loader))
                vocab_size = int(torch.max(sample_batch).item()) + 1
        
        if num_classes is None:
            # Get unique classes from the dataset
            all_labels = []
            for _, label in train_dataset:
                all_labels.append(label)
            num_classes = len(set(all_labels))
        
        # Build model if not already built
        if self.model is None:
            self.build_model(vocab_size, num_classes)
            
        # Training loop
        train_losses = []
        val_losses = []
        val_metrics = []
        best_val_loss = float('inf')
        
        print(f"Starting training with {len(train_dataset)} examples")
        print(f"Using device: {self.device}")
        
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Evaluate
            if val_loader:
                val_loss, val_preds, val_labels = self.evaluate(val_loader)
                val_losses.append(val_loss)
                
                # Calculate metrics
                accuracy = accuracy_score(val_labels, val_preds)
                precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
                recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
                f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
                
                val_metrics.append({
                    'loss': val_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(model_path)
                    print(f"Saved model with improved validation loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
                # Save model after each epoch without validation
                self.save_model(model_path)
        
        # Save training config
        config_path = os.path.splitext(model_path)[0] + '_config.json'
        
        config_data = {
            'config': self.config.to_dict(),
            'vocab_size': vocab_size,
            'num_classes': num_classes,
            'metrics': val_metrics[-1] if val_metrics else {}
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'best_val_loss': best_val_loss
        }
    
    def save_model(self, path: str) -> None:
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str, vocab_size: int = None, num_classes: int = None) -> None:
        """
        Load a model from a file.
        
        Args:
            path: Path to the model file
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
        """
        # Try to load config
        config_path = os.path.splitext(path)[0] + '_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update config
            self.config = TrainConfig(**config_data['config'])
            
            # Get vocab_size and num_classes from config if not provided
            if vocab_size is None:
                vocab_size = config_data.get('vocab_size')
            
            if num_classes is None:
                num_classes = config_data.get('num_classes')
        
        # Build model
        if vocab_size is not None and num_classes is not None:
            self.build_model(vocab_size, num_classes)
            
            # Load model state
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
        else:
            raise ValueError("Cannot load model: vocab_size and num_classes are required")