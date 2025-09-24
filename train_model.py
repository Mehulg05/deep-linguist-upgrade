import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(file_path):
    """Load sentiment analysis dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, max_words=10000, max_len=100):
    """Preprocess text data for training"""
    # Extract texts and labels
    texts = df['text'].values
    labels = df['sentiment'].values
    
    # Convert labels to binary (0 for negative, 1 for positive)
    labels = np.array([1 if label == 'positive' else 0 for label in labels])
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return X, labels, tokenizer

def create_gru_model(vocab_size, embedding_dim=100, max_len=100, gru_units=128):
    """Create GRU-based sentiment analysis model"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Dropout(0.2),
        
        # Bidirectional GRU layers
        Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(GRU(gru_units//2, dropout=0.2, recurrent_dropout=0.2)),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def train_model(X, y, tokenizer, epochs=50, batch_size=32, validation_split=0.2):
    """Train the GRU model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_gru_model(vocab_size, max_len=X.shape[1])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_gru_model.h5', save_best_only=True, monitor='val_accuracy')
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    
    # Save model and tokenizer
    model.save('sentiment_gru_model.h5')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Configuration
    MAX_WORDS = 10000
    MAX_LEN = 100
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Load data (replace with your dataset path)
    print("Loading data...")
    df = load_data('sentiment_data.csv')  # Replace with your dataset
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, tokenizer = preprocess_data(df, max_words=MAX_WORDS, max_len=MAX_LEN)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {np.sum(y)}")
    print(f"Negative samples: {len(y) - np.sum(y)}")
    
    # Train model
    print("Training GRU model...")
    model, history = train_model(X, y, tokenizer, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print("Training completed!")