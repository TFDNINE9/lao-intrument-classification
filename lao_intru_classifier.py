import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Parameters
SAMPLE_RATE = 44100
DURATION = 4  # seconds
DATA_PATH = "dataset"  # Update this to your dataset path
NUM_AUGMENTATIONS = 5  # Increased from 3 to 5 for more data diversity
MODEL_SAVE_PATH = "lao_instruments_model"

print("TensorFlow version:", tf.__version__)
print("Setting up Lao instrument classification pipeline...")

# Data Augmentation Functions
def augment_audio(audio, sr):
    """Apply random augmentations to audio sample"""
    augmented_samples = [audio]  # Keep original
    
    # Create specified number of augmentations
    for i in range(NUM_AUGMENTATIONS):
        aug_audio = audio.copy()
        
        # 1. Time stretching (0.85-1.15x speed)
        if random.random() > 0.5:
            stretch_factor = np.random.uniform(0.85, 1.15)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
        
        # 2. Pitch shifting (up to ±3 semitones) - Increased range
        if random.random() > 0.5:
            pitch_shift = np.random.uniform(-3, 3)
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=sr, n_steps=pitch_shift)
        
        # 3. Add small background noise (up to 8% of signal amplitude) - Increased slightly
        if random.random() > 0.5:
            noise_factor = np.random.uniform(0, 0.08)
            noise = np.random.randn(len(aug_audio))
            aug_audio = aug_audio + noise_factor * noise
        
        # 4. Time shifting (up to 25% of duration) - Increased range
        if random.random() > 0.5:
            shift_factor = int(np.random.uniform(-0.25, 0.25) * len(aug_audio))
            if shift_factor > 0:
                aug_audio = np.pad(aug_audio, (0, shift_factor), mode='constant')[:len(audio)]
            else:
                aug_audio = np.pad(aug_audio, (abs(shift_factor), 0), mode='constant')[abs(shift_factor):]
                if len(aug_audio) < len(audio):
                    aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        # 5. Volume adjustment (0.7-1.3x) - Increased range
        if random.random() > 0.5:
            volume_factor = np.random.uniform(0.7, 1.3)
            aug_audio = aug_audio * volume_factor
        
        # 6. Add random frequency filters (NEW)
        if random.random() > 0.7:  # Applied less frequently
            # Random bandpass filter
            low_cutoff = np.random.uniform(80, 500)
            high_cutoff = np.random.uniform(4000, 8000)
            aug_audio = librosa.effects.preemphasis(aug_audio, coef=0.97)
        
        # Ensure consistent length
        if len(aug_audio) > len(audio):
            aug_audio = aug_audio[:len(audio)]
        elif len(aug_audio) < len(audio):
            aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), mode='constant')
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

# Function to extract mel spectrograms
def extract_features(audio, sr):
    # Extract mel spectrogram with improved parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128,  # Keep this the same
        n_fft=2048,   # Increased for better frequency resolution
        hop_length=512,  # Decreased for better time resolution
        fmax=8000
    )
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Add delta features (time derivatives) to capture temporal dynamics
    delta = librosa.feature.delta(mel_spec_db)
    delta2 = librosa.feature.delta(mel_spec_db, order=2)
    
    # Stack features
    # Uncomment the following lines to use delta features
    # features = np.stack([mel_spec_db, delta, delta2], axis=-1)
    # return features
    
    # For now, return just the mel spectrogram to be compatible with your existing model
    return mel_spec_db

# Process all files and create dataset with augmentation
def create_dataset_with_augmentation():
    features = []
    labels = []
    file_paths = []  # Store original file paths for testing
    
    # Get all instrument classes
    instruments = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    print(f"Found instruments: {instruments}")
    
    for instrument in instruments:
        instrument_path = os.path.join(DATA_PATH, instrument)
        print(f"Processing {instrument}...")
        
        files = [f for f in os.listdir(instrument_path) if f.endswith('.wav')]
        print(f"  Found {len(files)} original audio files")
        
        for audio_file in files:
            file_path = os.path.join(instrument_path, audio_file)
            
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                
                # Ensure consistent length
                if len(audio) < DURATION * SAMPLE_RATE:
                    audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)))
                else:
                    audio = audio[:DURATION * SAMPLE_RATE]
                
                # Apply data augmentation
                augmented_samples = augment_audio(audio, sr)
                
                # Extract features for original and augmented samples
                for i, aug_audio in enumerate(augmented_samples):
                    mel_spec = extract_features(aug_audio, sr)
                    features.append(mel_spec)
                    labels.append(instrument)
                    
                    # Only store original file paths, not augmented ones
                    if i == 0:
                        file_paths.append(file_path)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    return np.array(features), np.array(labels), file_paths

# Define a smaller, more efficient CNN model
def build_improved_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),  # Light dropout after first block
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),  # Slight increase in dropout
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),  # More dropout
        
        # Fourth convolutional block (new)
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),  # Heavier dropout
        
        # Global pooling instead of flatten to reduce parameters
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Smaller dense layers
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Function to implement learning rate scheduling
def get_lr_scheduler():
    def lr_schedule(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.5
        else:
            return lr * 0.1
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Main execution
def main():
    print("Starting the Lao instrument classification pipeline...")
    
    # Create and prepare dataset with augmentation
    print("\n1. Extracting features with augmentation...")
    X, y, file_paths = create_dataset_with_augmentation()

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Print dataset information
    print(f"\nDataset shape after augmentation: {X.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}")

    # Split into training and testing sets (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Add channel dimension for CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    print(f"\n2. Building model...")
    # Create and compile the model
    model = build_improved_model(X_train.shape[1:], len(label_encoder.classes_))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Display model summary
    model.summary()

    print("\n3. Training model...")
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10, 
            restore_best_weights=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=5, 
            monitor='val_loss',
            min_lr=0.00001
        ),
        get_lr_scheduler()
    ]
    
    # Train the model with improved parameters
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Increase max epochs, early stopping will prevent overfitting
        batch_size=32,
        callbacks=callbacks
    )
    
    print("\n4. Evaluating model...")
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot learning rate
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    print("\n5. Saving model for Flutter integration...")
    # Save standard model
    model.save(f'{MODEL_SAVE_PATH}.h5')
    
    # Save the model in TensorFlow Lite format for mobile
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(f'{MODEL_SAVE_PATH}.tflite', 'wb') as f:
        f.write(tflite_model)

    # Save optimized version for better mobile performance
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    with open(f'{MODEL_SAVE_PATH}_quantized.tflite', 'wb') as f:
        f.write(tflite_quantized_model)

    # Save the label encoder classes
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder.classes_, f)

    print("Model and label encoder saved successfully!")
    
    print("\n6. Testing model on sample files...")
    # Test with a few samples
    test_samples = np.random.choice(file_paths, min(5, len(file_paths)), replace=False)
    
    # Modify the predict_instrument function to use the new model
    def predict_instrument(audio_path):
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Ensure consistent length
            if len(audio) < DURATION * SAMPLE_RATE:
                audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)))
            else:
                audio = audio[:DURATION * SAMPLE_RATE]
            
            # Extract features
            mel_spec = extract_features(audio, sr)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = model.predict(mel_spec)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]
            
            # Get instrument name
            instrument_name = label_encoder.classes_[predicted_class]
            
            return {
                'instrument': instrument_name,
                'confidence': float(confidence),
                'all_probabilities': {
                    instr: float(prob) for instr, prob in zip(label_encoder.classes_, prediction)
                }
            }
        except Exception as e:
            print(f"Error predicting {audio_path}: {str(e)}")
            return None
    
    # Test the model
    for test_file in test_samples:
        result = predict_instrument(test_file)
        if result:
            print(f"\nFile: {os.path.basename(test_file)}")
            print(f"Predicted: {result['instrument']} with {result['confidence']:.2%} confidence")
            print("All probabilities:")
            for instr, prob in result['all_probabilities'].items():
                print(f"  {instr}: {prob:.2%}")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()