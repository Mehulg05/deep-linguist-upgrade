import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class SentimentPredictor:
    def __init__(self, model_path='sentiment_gru_model.h5', tokenizer_path='tokenizer.pickle'):
        """Initialize the sentiment predictor with trained GRU model"""
        self.model = load_model(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        self.max_len = 100  # Should match training configuration
        
    def preprocess_text(self, text):
        """Preprocess single text for prediction"""
        if isinstance(text, str):
            text = [text]
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(text)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict(processed_text)[0][0]
        
        sentiment = 'positive' if prediction > 0.5 else 'negative'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'score': float(prediction)
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        processed_texts = self.preprocess_text(texts)
        predictions = self.model.predict(processed_texts)
        
        results = []
        for i, text in enumerate(texts):
            score = predictions[i][0]
            sentiment = 'positive' if score > 0.5 else 'negative'
            confidence = score if score > 0.5 else 1 - score
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': float(confidence),
                'score': float(score)
            })
        
        return results

def main():
    """Example usage of the sentiment predictor"""
    # Initialize predictor
    predictor = SentimentPredictor()
    
    # Example texts
    sample_texts = [
        "I love this movie! It's absolutely fantastic!",
        "This product is terrible and I hate it.",
        "The weather is okay today.",
        "Amazing service and great quality!",
        "I'm not sure about this decision.",
        "This is the worst experience ever.",
        "Pretty good overall, I'm satisfied.",
        "Neutral opinion, nothing special."
    ]
    
    print("GRU Sentiment Analysis Predictions:")
    print("=" * 50)
    
    # Single predictions
    for text in sample_texts:
        result = predictor.predict_sentiment(text)
        print(f"Text: {result['text'][:50]}...")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Score: {result['score']:.4f}")
        print("-" * 50)
    
    # Batch prediction
    print("\nBatch Prediction Results:")
    batch_results = predictor.predict_batch(sample_texts)
    
    for result in batch_results:
        print(f"{result['sentiment'].upper()}: {result['confidence']:.3f} - {result['text'][:30]}...")

if __name__ == "__main__":
    main()