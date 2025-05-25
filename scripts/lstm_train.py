
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Layer, Softmax, Multiply
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from custom_layer import Attention
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    X = [item["sequence"] for item in data]
    y = [item["label"] for item in data]
    # Pad sequences (right padding)
    X_padded = pad_sequences(X, padding='post', dtype='float32')
    y = np.array(y, dtype=np.float32)
    return X_padded, y

def build_model(timesteps, features):
    input_layer = Input(shape=(timesteps, features))
    # x = Masking(mask_value=0.0)(input_layer)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = Attention()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    X_padded, y = load_and_preprocess_data("bowling_sequences.json")
    print(f"Data shapes: X={X_padded.shape}, y={y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    # Check if any non-zero values after first zero in each sequence along time axis
    for i, seq in enumerate(X_padded):
        zero_pos = np.where(np.all(seq == 0, axis=1))[0]
        if len(zero_pos) > 0:
            first_zero = zero_pos[0]
            if np.any(seq[first_zero+1:] != 0):
                print(f"Sequence {i} has non-zero values after zeros at timestep {first_zero}")

    # Build model
    early_stopping = EarlyStopping(
        monitor='val_loss',      # You can also use 'val_accuracy'
        patience=5,             
        restore_best_weights=True
    )
    model = build_model(X_padded.shape[1], X_padded.shape[2])
    model.summary()

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping])

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")

    with open("training_history.json", "w") as f:
        json.dump(history.history, f)

    # Save model
    model.save("bowling_lstm_classifier.keras")
    print("Model saved as bowling_lstm_classifier.keras")
