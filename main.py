#!/usr/bin/env python3
"""
Sentiment Analysis with GRU Neural Network
Main script to train model or make predictions
"""

import argparse
import os
import sys
from train_model import load_data, preprocess_data, train_model
from predict import SentimentPredictor
from model_utils import ModelEvaluator, compare_models_performance

def train_new_model(data_path, epochs=50, batch_size=32):
    """Train a new GRU sentiment analysis model"""
    print("Starting GRU model training...")
    
    # Load and preprocess data
    df = load_data(data_path)
    X, y, tokenizer = preprocess_data(df, max_words=10000, max_len=100)
    
    # Train model
    model, history = train_model(X, y, tokenizer, epochs=epochs, batch_size=batch_size)
    
    print("Model training completed successfully!")
    print("Saved files:")
    print("- sentiment_gru_model.h5 (trained model)")
    print("- tokenizer.pickle (text tokenizer)")
    print("- best_gru_model.h5 (best model checkpoint)")
    print("- training_history.png (training visualization)")

def predict_sentiment(text_input):
    """Make sentiment prediction on input text"""
    if not os.path.exists('sentiment_gru_model.h5'):
        print("Error: No trained model found. Please train a model first.")
        return
    
    predictor = SentimentPredictor()
    
    if os.path.isfile(text_input):
        # Read from file
        with open(text_input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Analyzing {len(texts)} texts from file...")
        results = predictor.predict_batch(texts)
        
        for result in results:
            print(f"Text: {result['text'][:60]}...")
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 50)
    else:
        # Single text prediction
        result = predictor.predict_sentiment(text_input)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Score: {result['score']:.4f}")

def evaluate_model(test_data_path):
    """Evaluate trained model on test data"""
    if not os.path.exists('sentiment_gru_model.h5'):
        print("Error: No trained model found. Please train a model first.")
        return
    
    # Load test data
    df = load_data(test_data_path)
    X_test, y_test, _ = preprocess_data(df, max_words=10000, max_len=100)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(X_test, y_test)
    
    print("Evaluation completed!")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='GRU Sentiment Analysis Tool')
    parser.add_argument('mode', choices=['train', 'predict', 'evaluate', 'compare'], 
                       help='Operation mode')
    parser.add_argument('--data', type=str, help='Path to dataset (CSV file)')
    parser.add_argument('--text', type=str, help='Text to analyze or file path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data:
            print("Error: --data argument required for training")
            sys.exit(1)
        train_new_model(args.data, args.epochs, args.batch_size)
    
    elif args.mode == 'predict':
        if not args.text:
            print("Error: --text argument required for prediction")
            sys.exit(1)
        predict_sentiment(args.text)
    
    elif args.mode == 'evaluate':
        if not args.data:
            print("Error: --data argument required for evaluation")
            sys.exit(1)
        evaluate_model(args.data)
    
    elif args.mode == 'compare':
        compare_models_performance()

if __name__ == "__main__":
    # If no arguments provided, show interactive menu
    if len(sys.argv) == 1:
        print("GRU Sentiment Analysis Tool")
        print("=" * 30)
        print("1. Train new model")
        print("2. Make prediction")
        print("3. Evaluate model")
        print("4. Compare GRU vs LSTM")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            data_path = input("Enter path to training data (CSV): ").strip()
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            train_new_model(data_path, epochs)
        
        elif choice == '2':
            text = input("Enter text to analyze: ").strip()
            predict_sentiment(text)
        
        elif choice == '3':
            data_path = input("Enter path to test data (CSV): ").strip()
            evaluate_model(data_path)
        
        elif choice == '4':
            compare_models_performance()
        
        elif choice == '5':
            print("Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice!")
            sys.exit(1)
    
    else:
        main()