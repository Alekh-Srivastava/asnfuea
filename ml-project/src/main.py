
"""
Main entry point for the sentiment analysis pipeline with GRU.
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Any

from src.data.processor import TextProcessor, DataConfig
from src.models.trainer import SentimentTrainer, TrainConfig
from src.models.inference import SentimentPredictor, InferenceConfig


def train_pipeline(args):
    """Run the training pipeline."""
    print(f"Loading data from {args.data_path}")
    
    try:
        # Load data
        df = pd.read_csv(args.data_path, encoding='latin-1')
        
        # Set up data processor
        data_config = DataConfig(
            max_sequence_length=args.max_seq_length,
            max_vocab_size=args.max_vocab_size,
            embedding_dim=args.embedding_dim
        )
        processor = TextProcessor(config=data_config)
        
        # Identify text columns for COVID dataset or general comments
        if 'OriginalTweet' in df.columns:
            # COVID-19 Twitter dataset
            text_col = 'OriginalTweet'
            print(f"Found COVID Twitter dataset with '{text_col}' column")
            # Create a single text column
            df['cleaned_text'] = df[text_col].apply(processor._clean_text)
        else:
            # Try to find comment columns
            comment_cols = [col for col in df.columns if col.startswith('Comment')]
            if not comment_cols:
                raise ValueError("No OriginalTweet or Comment columns found in the data.")
            print(f"Found {len(comment_cols)} comment columns: {comment_cols}")
            # Process data by combining comments
            df = processor.combine_comments(df, comment_cols)
        
        # Split into train and validation
        train_df, val_df = train_test_split(
            df, 
            test_size=data_config.test_size, 
            random_state=data_config.random_state
        )
        
        # Prepare text data for the model
        train_texts = train_df['cleaned_text'].tolist()
        val_texts = val_df['cleaned_text'].tolist()
        
        # Handle labels based on available columns
        if 'Sentiment' in train_df.columns:
            # COVID Twitter dataset uses 'Sentiment' column
            # Map sentiment labels (Extremely Negative, Negative, Neutral, Positive, Extremely Positive) to (0, 1, 2)
            sentiment_map = {
                'Extremely Negative': 0, 'Negative': 0,
                'Neutral': 1,
                'Positive': 2, 'Extremely Positive': 2
            }
            train_df['sentiment'] = train_df['Sentiment'].map(sentiment_map)
            val_df['sentiment'] = val_df['Sentiment'].map(sentiment_map)
            train_labels = train_df['sentiment'].values
            val_labels = val_df['sentiment'].values
            print(f"Using 'Sentiment' column from COVID dataset. Mapped to 3 classes.")
        elif 'sentiment' in train_df.columns:
            # Use existing sentiment column
            train_labels = train_df['sentiment'].values
            val_labels = val_df['sentiment'].values
            print(f"Using existing 'sentiment' column.")
        else:
            # Create dummy sentiment labels
            np.random.seed(42)
            train_labels = np.random.randint(0, 3, size=len(train_df))
            val_labels = np.random.randint(0, 3, size=len(val_df))
            print(f"No sentiment column found. Using random labels for demonstration.")
        
        # Create datasets
        datasets = processor.create_datasets(train_texts, train_labels, val_texts, val_labels)
        
        # Save vocabulary
        os.makedirs('models', exist_ok=True)
        vocab_path = os.path.join('models', 'vocab.pkl')
        processor.save_vocab(vocab_path)
        
        # Set up model trainer
        train_config = TrainConfig(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        
        trainer = SentimentTrainer(config=train_config)
        
        # Train model
        model_path = os.path.join('models', f"{args.model_name}.pt")
        results = trainer.train(
            datasets['train_dataset'],
            datasets['val_dataset'],
            datasets['vocab_size'],
            num_classes=len(np.unique(train_labels)),
            model_path=model_path
        )
        
        # Save label mapping
        config_path = os.path.splitext(model_path)[0] + '_config.json'
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config_data = {'config': train_config.to_dict()}
        
        # Add label mapping
        label_map = {}
        for i, label in enumerate(np.unique(train_labels)):
            label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(i, f"Class {i}")
            label_map[str(i)] = label_name
        
        config_data['label_map'] = label_map
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        # Plot training history
        if 'val_losses' in results and results['val_losses']:
            plt.figure(figsize=(10, 6))
            plt.plot(results['train_losses'], label='Training Loss')
            plt.plot(results['val_losses'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_history.png')
            
            # Print final metrics
            if 'val_metrics' in results and results['val_metrics']:
                final_metrics = results['val_metrics'][-1]
                print(f"Training complete. Model saved to {model_path}")
                print(f"Final validation metrics:")
                print(f"  Loss: {final_metrics.get('loss', 'N/A')}")
                print(f"  Accuracy: {final_metrics.get('accuracy', 'N/A')}")
                print(f"  F1 Score: {final_metrics.get('f1', 'N/A')}")
            else:
                print(f"Training complete. Model saved to {model_path}")
        else:
            print(f"Training complete. Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        raise


def inference_pipeline(args):
    """Run the inference pipeline."""
    try:
        # Load data
        df = pd.read_csv(args.data_path, encoding='latin-1')
        
        # Set up predictor
        inference_config = InferenceConfig(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size
        )
        predictor = SentimentPredictor(config=inference_config)
        
        # Identify text columns for COVID dataset or general comments
        if 'OriginalTweet' in df.columns:
            # COVID-19 Twitter dataset
            print(f"Found COVID Twitter dataset with 'OriginalTweet' column")
            # Create a cleaned text column
            df['cleaned_text'] = df['OriginalTweet'].apply(predictor._clean_text)
            text_col = 'cleaned_text'
        else:
            # Try to find comment columns
            comment_cols = [col for col in df.columns if col.startswith('Comment')]
            if not comment_cols:
                if 'text' in df.columns:
                    text_col = 'text'
                else:
                    raise ValueError("No OriginalTweet, Comment, or text columns found in the data.")
            else:
                # Combine comments
                processor = TextProcessor()
                df = processor.combine_comments(df, comment_cols)
                text_col = 'cleaned_text'
        
        # Make predictions
        result_df = predictor.predict_dataframe(df, text_col)
        
        # Save results
        output_path = args.output_path or 'data/processed/sentiment_predictions.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        
        print(f"Inference complete. Results saved to {output_path}")
        
        # Display sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment_label', data=result_df)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png')
        
    except Exception as e:
        print(f"Error in inference pipeline: {e}")
        raise


def main():
    """Main entry point for the pipeline."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline with GRU')
    
    # Add shared arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/inference')
    
    # Create subparsers for train and inference modes
    subparsers = parser.add_subparsers(dest='mode', help='Pipeline mode')
    
    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train a sentiment model')
    train_parser.add_argument('--model_name', type=str, default='sentiment_gru', help='Name for the saved model')
    train_parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum sequence length for text')
    train_parser.add_argument('--max_vocab_size', type=int, default=10000, help='Maximum vocabulary size')
    train_parser.add_argument('--embedding_dim', type=int, default=194, help='Dimension of word embeddings')
    train_parser.add_argument('--hidden_dim', type=int, default=243, help='Dimension of hidden layer in GRU')
    train_parser.add_argument('--n_layers', type=int, default=3, help='Number of GRU layers')
    train_parser.add_argument('--dropout_rate', type=float, default=0.104, help='Dropout rate')
    train_parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    train_parser.add_argument('--learning_rate', type=float, default=0.0032, help='Learning rate')
    
    # Inference arguments
    inference_parser = subparsers.add_parser('inference', help='Make predictions with a trained model')
    inference_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    inference_parser.add_argument('--vocab_path', type=str, required=True, help='Path to the saved vocabulary')
    inference_parser.add_argument('--output_path', type=str, help='Path to save the predictions')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate pipeline
    if args.mode == 'train':
        train_pipeline(args)
    elif args.mode == 'inference':
        inference_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()