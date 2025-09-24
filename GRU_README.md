# Sentiment Analysis with GRU Neural Network

A Python implementation of sentiment analysis using Gated Recurrent Unit (GRU) neural networks. This project provides a complete pipeline for training, evaluating, and using GRU models for text sentiment classification.

## Features

- **GRU-based Architecture**: Uses efficient GRU layers instead of LSTM for faster training
- **Bidirectional Processing**: Implements bidirectional GRU for better context understanding  
- **Complete Pipeline**: Training, evaluation, and prediction modules
- **Model Evaluation**: Comprehensive metrics including ROC curves and confusion matrices
- **Batch Processing**: Support for single text and batch predictions
- **Interactive CLI**: Command-line interface for easy usage

## Model Architecture

```
Input Text → Tokenization → Embedding Layer → Dropout
     ↓
Bidirectional GRU (128 units) → Bidirectional GRU (64 units)
     ↓
Dense (64) → Dropout → Dense (32) → Dropout → Dense (1, Sigmoid)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sentiment-analysis-gru
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

**Train a new model:**
```bash
python main.py train --data your_dataset.csv --epochs 50 --batch-size 32
```

**Make predictions:**
```bash
python main.py predict --text "This movie is amazing!"
```

**Evaluate model:**
```bash
python main.py evaluate --data test_dataset.csv
```

**Compare GRU vs LSTM:**
```bash
python main.py compare
```

### Interactive Mode

Run without arguments for interactive menu:
```bash
python main.py
```

### Python API

**Training:**
```python
from train_model import load_data, preprocess_data, train_model

# Load and preprocess data
df = load_data('dataset.csv')
X, y, tokenizer = preprocess_data(df)

# Train model
model, history = train_model(X, y, tokenizer)
```

**Prediction:**
```python
from predict import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor()

# Single prediction
result = predictor.predict_sentiment("I love this product!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")

# Batch prediction
texts = ["Great movie!", "Terrible service", "Average quality"]
results = predictor.predict_batch(texts)
```

## Data Format

Your dataset should be a CSV file with the following columns:
- `text`: The text content to analyze
- `sentiment`: The sentiment label ('positive' or 'negative')

Example:
```csv
text,sentiment
"I love this movie!",positive
"This product is terrible",negative
"Average quality product",neutral
```

## Model Files

After training, the following files are generated:
- `sentiment_gru_model.h5`: Main trained model
- `tokenizer.pickle`: Text tokenizer for preprocessing
- `best_gru_model.h5`: Best model checkpoint during training
- `training_history.png`: Training visualization
- `confusion_matrix.png`: Model evaluation metrics
- `roc_curve.png`: ROC curve analysis

## GRU vs LSTM

This implementation uses GRU instead of LSTM for the following advantages:

### GRU Benefits:
- **Fewer Parameters**: 25% fewer parameters than LSTM
- **Faster Training**: Reduced computational complexity
- **Less Overfitting**: Simpler architecture reduces overfitting risk
- **Similar Performance**: Comparable accuracy to LSTM on most tasks

### Technical Differences:
- **Gates**: GRU uses 2 gates (reset, update) vs LSTM's 3 gates (input, forget, output)
- **Memory**: Simplified memory mechanism
- **Computation**: Faster forward and backward propagation

## Requirements

- Python 3.7+
- TensorFlow 2.12+
- NumPy
- Pandas  
- Scikit-learn
- Matplotlib
- Seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.