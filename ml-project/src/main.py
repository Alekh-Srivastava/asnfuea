import argparse
import pandas as pd
from data.processor import process_data
from models.trainer import train_sentiment_model, TrainConfig
from models.inference import SentimentPredictor, InferenceConfig, add_sentiment_prediction

def main(args):
    """Main function to run the ML pipeline."""
    print("--- ML Pipeline Started ---")
    
    if args.mode == 'train':
        print(f"Running in TRAIN mode for data at: {args.data_path}")
        raw_df = pd.read_csv(args.data_path)
        # For performance, you might want to train on a smaller sample
        # raw_df = raw_df.sample(n=10000, random_state=42) 
        processed_df = process_data(raw_df)
        
        config = TrainConfig(epochs=args.epochs, learning_rate=args.lr)
        train_sentiment_model(processed_df, model_save_path=args.model_path, config=config)
        
    elif args.mode == 'predict':
        print(f"Running in PREDICT mode for data at: {args.data_path}")
        raw_df = pd.read_csv(args.data_path)
        
        infer_config = InferenceConfig(model_path=args.model_path)
        predictor = SentimentPredictor(config=infer_config)
        
        results_df = add_sentiment_prediction(raw_df, predictor)
        
        output_path = "data/processed/predictions.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
        print("\nResults Sample:")
        print(results_df[['review', 'sentiment', 'predicted_sentiment']].head())
        
    else:
        print(f"❌ Error: Unknown mode '{args.mode}'. Choose 'train' or 'predict'.")

    print("--- ML Pipeline Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the sentiment analysis ML pipeline.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help="The mode to run the pipeline in: 'train' or 'predict'.")
    # After
    parser.add_argument('--data_path', type=str, default='/Users/nafeessiddiqui/Desktop/mock/ml-project/data/raw/IMDB Dataset.csv',
                    help="Path to the input data CSV file.")
    parser.add_argument('--model_path', type=str, default='models/sentiment_cnn.pth',
                        help="Path to save or load the model.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training.")

    args = parser.parse_args()
    main(args)