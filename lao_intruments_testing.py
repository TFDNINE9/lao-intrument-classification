import librosa
import numpy as np
import tensorflow as tf
import os
import pickle

# Parameters
SAMPLE_RATE = 44100
DURATION = 4  # seconds
MODEL_SAVE_PATH = "lao_instruments_model"

def extract_features(audio, sr):
    # Extract mel spectrogram with improved parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmax=8000
    )
    
    # Convert to decibels
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # For now, return just the mel spectrogram to be compatible with your existing model
    return mel_spec_db

def test_model_on_audio_files(model_path, label_encoder_path, audio_files):
    """
    Test the saved model on a list of audio files.
    
    Parameters:
    model_path (str): Path to the saved model (.h5 file)
    label_encoder_path (str): Path to the saved label encoder (.pkl file)
    audio_files (list): List of audio file paths to test
    
    Returns:
    list: List of dictionaries containing prediction results for each file
    """
    print("Loading model and label encoder...")
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load the label encoder classes
    with open(label_encoder_path, 'rb') as f:
        classes = pickle.load(f)
    
    results = []
    
    print(f"Testing {len(audio_files)} audio samples...")
    
    # Test each file
    for audio_path in audio_files:
        try:
            print(f"\nProcessing file: {os.path.basename(audio_path)}")
            
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
            instrument_name = classes[predicted_class]
            
            result = {
                'file': os.path.basename(audio_path),
                'instrument': instrument_name,
                'confidence': float(confidence),
                'all_probabilities': {
                    instr: float(prob) for instr, prob in zip(classes, prediction)
                }
            }
            
            results.append(result)
            
            # Print result
            print(f"Predicted: {result['instrument']} with {result['confidence']:.2%} confidence")
            print("All probabilities:")
            for instr, prob in result['all_probabilities'].items():
                print(f"  {instr}: {prob:.2%}")
                
        except Exception as e:
            print(f"Error predicting {audio_path}: {str(e)}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Paths to model and label encoder
    model_path = f"{MODEL_SAVE_PATH}.h5"  # or use the TFLite version if needed
    label_encoder_path = "label_encoder.pkl"
    
    # Example: Test with specific audio files
    audio_files = [
        "testing_data/pin_test-01.wav",
        "testing_data/pin_test-02.wav",
        "testing_data/pin_test-03.wav",
        "testing_data/pin_test-04.wav",
        "testing_data/pin_test-05.wav",
        "testing_data/pin_test-06.wav",
        "testing_data/pin_test-07.wav",
    ]
    
    # Or test with files from a directory
    """
    test_dir = "path/to/test_folder"
    audio_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.wav')]
    """
    
    results = test_model_on_audio_files(model_path, label_encoder_path, audio_files)
    
    # You can also save the results to a file if needed
    """
    import json
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    """