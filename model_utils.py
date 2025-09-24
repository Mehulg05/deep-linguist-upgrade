import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model
import pickle

class ModelEvaluator:
    def __init__(self, model_path='sentiment_gru_model.h5', tokenizer_path='tokenizer.pickle'):
        """Initialize model evaluator"""
        self.model = load_model(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        
        # ROC curve
        self.plot_roc_curve(y_test, y_pred_prob)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - GRU Sentiment Analysis')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - GRU Sentiment Analysis')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.show()
    
    def analyze_predictions(self, texts, y_true, y_pred, y_pred_prob, num_samples=10):
        """Analyze specific predictions"""
        # Get indices for correct and incorrect predictions
        correct_indices = np.where(y_true == y_pred.flatten())[0]
        incorrect_indices = np.where(y_true != y_pred.flatten())[0]
        
        print(f"\nCorrect Predictions: {len(correct_indices)}")
        print(f"Incorrect Predictions: {len(incorrect_indices)}")
        
        # Show some examples
        print(f"\nSample Correct Predictions:")
        for i in correct_indices[:num_samples]:
            sentiment = 'Positive' if y_true[i] == 1 else 'Negative'
            confidence = y_pred_prob[i][0] if y_true[i] == 1 else 1 - y_pred_prob[i][0]
            print(f"Text: {texts[i][:60]}...")
            print(f"True/Predicted: {sentiment}, Confidence: {confidence:.3f}\n")
        
        print(f"Sample Incorrect Predictions:")
        for i in incorrect_indices[:num_samples]:
            true_sentiment = 'Positive' if y_true[i] == 1 else 'Negative'
            pred_sentiment = 'Positive' if y_pred[i] == 1 else 'Negative'
            confidence = y_pred_prob[i][0] if y_pred[i] == 1 else 1 - y_pred_prob[i][0]
            print(f"Text: {texts[i][:60]}...")
            print(f"True: {true_sentiment}, Predicted: {pred_sentiment}, Confidence: {confidence:.3f}\n")

def compare_models_performance():
    """Compare GRU vs LSTM performance (theoretical comparison)"""
    print("GRU vs LSTM Comparison:")
    print("=" * 40)
    print("GRU Advantages:")
    print("- Fewer parameters (faster training)")
    print("- Less prone to overfitting")
    print("- Better performance on smaller datasets")
    print("- Simpler architecture")
    print("\nGRU Characteristics:")
    print("- 2 gates (reset and update) vs LSTM's 3 gates")
    print("- Combines forget and input gates into update gate")
    print("- Generally faster computation")
    print("- Similar performance to LSTM on most tasks")

def model_summary_comparison():
    """Display model architecture summary"""
    print("GRU Model Architecture:")
    print("=" * 30)
    print("1. Embedding Layer")
    print("2. Dropout (0.2)")
    print("3. Bidirectional GRU (128 units, return_sequences=True)")
    print("4. Bidirectional GRU (64 units)")
    print("5. Dense (64 units, ReLU)")
    print("6. Dropout (0.5)")
    print("7. Dense (32 units, ReLU)")
    print("8. Dropout (0.3)")
    print("9. Dense (1 unit, Sigmoid)")

if __name__ == "__main__":
    compare_models_performance()
    model_summary_comparison()