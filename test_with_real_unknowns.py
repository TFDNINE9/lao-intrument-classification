import numpy as np
import librosa
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import soundfile as sf  # For saving audio files

# Parameters
SAMPLE_RATE = 44100
DURATION = 4  # seconds
MODEL_SAVE_PATH = "lao_instruments_model"

def extract_features(audio, sr):
    """Extract Mel spectrogram features from audio"""
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
    return mel_spec_db

def predict_instrument_with_unknown_detection(audio_path, model, label_encoder, 
                                             confidence_threshold=0.9, entropy_threshold=0.12):
    """
    Predict instrument with unknown detection using both confidence and entropy thresholds
    
    Parameters:
    audio_path (str): Path to the audio file
    model: Trained model
    label_encoder: Label encoder with class names
    confidence_threshold: Minimum confidence to accept prediction
    entropy_threshold: Maximum normalized entropy to accept prediction
    
    Returns:
    dict: Prediction results with possible 'Unknown' label
    """
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
        prediction = model.predict(mel_spec, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Calculate entropy
        entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
        max_entropy = np.log2(len(prediction))
        normalized_entropy = entropy / max_entropy
        
        # Calculate dispersion (gap between highest and second highest prediction)
        sorted_pred = np.sort(prediction)[::-1]  # Sort in descending order
        if len(sorted_pred) > 1:
            top_gap = sorted_pred[0] - sorted_pred[1]
        else:
            top_gap = 1.0
        
        # Determine if the sound is unknown based on multiple criteria
        is_unknown = confidence < confidence_threshold or normalized_entropy > entropy_threshold
        
        # Get instrument name
        if is_unknown:
            instrument_name = "Unknown"
        else:
            instrument_name = label_encoder.classes_[predicted_class]
        
        return {
            'instrument': instrument_name,
            'confidence': float(confidence),
            'entropy': float(normalized_entropy),
            'top_gap': float(top_gap),
            'confidence_check': confidence >= confidence_threshold,
            'entropy_check': normalized_entropy <= entropy_threshold,
            'is_unknown': is_unknown,
            'all_probabilities': {
                instr: float(prob) for instr, prob in zip(label_encoder.classes_, prediction)
            }
        }
    except Exception as e:
        print(f"Error predicting {audio_path}: {str(e)}")
        return None

def test_unknown_detection(model, label_encoder, known_samples, unknown_samples, threshold_config=None):
    """
    Test unknown detection system and visualize results
    
    Parameters:
    model: Trained model
    label_encoder: Label encoder
    known_samples: List of paths to known instrument samples
    unknown_samples: List of paths to unknown instrument samples
    threshold_config: Dictionary with confidence_threshold and entropy_threshold
    
    Returns:
    dict: Stats about detection performance
    """
    if threshold_config is None:
        threshold_config = {
            'confidence_threshold': 0.9,
            'entropy_threshold': 0.12
        }
    
    # Collect stats
    results = {
        'known': {
            'total': len(known_samples),
            'correctly_identified': 0,
            'wrongly_marked_unknown': 0,
            'confidences': [],
            'entropies': []
        },
        'unknown': {
            'total': len(unknown_samples),
            'correctly_identified_unknown': 0,
            'wrongly_classified': 0,
            'confidences': [],
            'entropies': []
        }
    }
    
    # Process known samples
    print("\nTesting with known instrument samples:")
    for sample_path in known_samples:
        result = predict_instrument_with_unknown_detection(
            sample_path, model, label_encoder,
            confidence_threshold=threshold_config['confidence_threshold'],
            entropy_threshold=threshold_config['entropy_threshold']
        )
        
        if result:
            print(f"File: {os.path.basename(sample_path)}")
            print(f"Predicted: {result['instrument']} (Confidence: {result['confidence']:.2f}, Entropy: {result['entropy']:.2f})")
            
            results['known']['confidences'].append(result['confidence'])
            results['known']['entropies'].append(result['entropy'])
            
            if result['instrument'] != "Unknown":
                results['known']['correctly_identified'] += 1
            else:
                results['known']['wrongly_marked_unknown'] += 1
            
            print()
    
    # Process unknown samples
    print("\nTesting with unknown instrument samples:")
    for sample_path in unknown_samples:
        result = predict_instrument_with_unknown_detection(
            sample_path, model, label_encoder,
            confidence_threshold=threshold_config['confidence_threshold'],
            entropy_threshold=threshold_config['entropy_threshold']
        )
        
        if result:
            print(f"File: {os.path.basename(sample_path)}")
            print(f"Predicted: {result['instrument']} (Confidence: {result['confidence']:.2f}, Entropy: {result['entropy']:.2f})")
            
            results['unknown']['confidences'].append(result['confidence'])
            results['unknown']['entropies'].append(result['entropy'])
            
            if result['instrument'] == "Unknown":
                results['unknown']['correctly_identified_unknown'] += 1
            else:
                results['unknown']['wrongly_classified'] += 1
            
            print()
    
    # Calculate performance metrics
    known_accuracy = results['known']['correctly_identified'] / results['known']['total'] if results['known']['total'] > 0 else 0
    unknown_accuracy = results['unknown']['correctly_identified_unknown'] / results['unknown']['total'] if results['unknown']['total'] > 0 else 0
    overall_accuracy = (results['known']['correctly_identified'] + results['unknown']['correctly_identified_unknown']) / (results['known']['total'] + results['unknown']['total'])
    
    print("\nPerformance Summary:")
    print(f"Known instruments correctly identified: {results['known']['correctly_identified']}/{results['known']['total']} ({known_accuracy:.2%})")
    print(f"Unknown sounds correctly marked as Unknown: {results['unknown']['correctly_identified_unknown']}/{results['unknown']['total']} ({unknown_accuracy:.2%})")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    # Visualize confidence and entropy distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if results['known']['confidences']:
        plt.hist(results['known']['confidences'], alpha=0.5, label='Known Instruments')
    if results['unknown']['confidences']:
        plt.hist(results['unknown']['confidences'], alpha=0.5, label='Unknown Sounds')
    plt.axvline(x=threshold_config['confidence_threshold'], color='r', linestyle='--', label=f'Threshold ({threshold_config["confidence_threshold"]})')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if results['known']['entropies']:
        plt.hist(results['known']['entropies'], alpha=0.5, label='Known Instruments')
    if results['unknown']['entropies']:
        plt.hist(results['unknown']['entropies'], alpha=0.5, label='Unknown Sounds')
    plt.axvline(x=threshold_config['entropy_threshold'], color='r', linestyle='--', label=f'Threshold ({threshold_config["entropy_threshold"]})')
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Count')
    plt.title('Entropy Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('unknown_detection_performance.png')
    plt.show()
    
    return results

def visualize_distributions(model, label_encoder, known_samples, unknown_samples):
    """
    Visualize confidence and entropy distributions for manual threshold tuning
    
    Parameters:
    model: Trained model
    label_encoder: Label encoder
    known_samples: List of paths to known instrument samples
    unknown_samples: List of paths to unknown instrument samples
    """
    # Collect data
    known_confidences = []
    known_entropies = []
    unknown_confidences = []
    unknown_entropies = []
    
    # Process known samples
    print("Analyzing known instrument samples...")
    for sample_path in known_samples:
        try:
            # Load audio
            audio, sr = librosa.load(sample_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Ensure consistent length
            if len(audio) < DURATION * SAMPLE_RATE:
                audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)))
            else:
                audio = audio[:DURATION * SAMPLE_RATE]
            
            # Extract features
            mel_spec = extract_features(audio, sr)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = model.predict(mel_spec, verbose=0)[0]
            confidence = np.max(prediction)
            
            # Calculate entropy
            entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
            max_entropy = np.log2(len(prediction))
            normalized_entropy = entropy / max_entropy
            
            known_confidences.append(confidence)
            known_entropies.append(normalized_entropy)
            
            # Print out for each file
            print(f"File: {os.path.basename(sample_path)}, "
                  f"Confidence: {confidence:.2f}, Entropy: {normalized_entropy:.2f}")
            
        except Exception as e:
            print(f"Error processing {sample_path}: {str(e)}")
    
    # Process unknown samples
    print("\nAnalyzing unknown instrument samples...")
    for sample_path in unknown_samples:
        try:
            # Load audio
            audio, sr = librosa.load(sample_path, sr=SAMPLE_RATE, duration=DURATION)
            
            # Ensure consistent length
            if len(audio) < DURATION * SAMPLE_RATE:
                audio = np.pad(audio, (0, DURATION * SAMPLE_RATE - len(audio)))
            else:
                audio = audio[:DURATION * SAMPLE_RATE]
            
            # Extract features
            mel_spec = extract_features(audio, sr)
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            
            # Make prediction
            prediction = model.predict(mel_spec, verbose=0)[0]
            confidence = np.max(prediction)
            
            # Calculate entropy
            entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
            max_entropy = np.log2(len(prediction))
            normalized_entropy = entropy / max_entropy
            
            unknown_confidences.append(confidence)
            unknown_entropies.append(normalized_entropy)
        except Exception as e:
            print(f"Error processing {sample_path}: {str(e)}")
    
    # Visualize distributions
    plt.figure(figsize=(15, 10))
    
    # Confidence histogram
    plt.subplot(2, 2, 1)
    plt.hist(known_confidences, bins=20, alpha=0.5, label='Known Instruments')
    plt.hist(unknown_confidences, bins=20, alpha=0.5, label='Unknown Instruments')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # Entropy histogram
    plt.subplot(2, 2, 2)
    plt.hist(known_entropies, bins=20, alpha=0.5, label='Known Instruments')
    plt.hist(unknown_entropies, bins=20, alpha=0.5, label='Unknown Instruments')
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Count')
    plt.title('Entropy Distribution')
    plt.legend()
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(known_entropies, known_confidences, alpha=0.7, label='Known Instruments')
    plt.scatter(unknown_entropies, unknown_confidences, alpha=0.7, label='Unknown Instruments')
    plt.xlabel('Normalized Entropy')
    plt.ylabel('Confidence')
    plt.title('Confidence vs. Entropy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add lines for potential thresholds
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Confidence Threshold = 0.9')
    plt.axvline(x=0.12, color='g', linestyle='--', alpha=0.5, label='Entropy Threshold = 0.12')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('manual_threshold_visualization.png')
    plt.show()
    
    # Calculate performance metrics for suggested thresholds
    conf_thresh = 0.9
    entr_thresh = 0.12
    
    known_correct = 0
    unknown_correct = 0
    
    for conf, entr in zip(known_confidences, known_entropies):
        is_unknown = conf < conf_thresh or entr > entr_thresh
        if not is_unknown:  # Correctly identified as known
            known_correct += 1
    
    for conf, entr in zip(unknown_confidences, unknown_entropies):
        is_unknown = conf < conf_thresh or entr > entr_thresh
        if is_unknown:  # Correctly identified as unknown
            unknown_correct += 1
    
    known_accuracy = known_correct / len(known_confidences) if known_confidences else 0
    unknown_accuracy = unknown_correct / len(unknown_confidences) if unknown_confidences else 0
    overall_accuracy = (known_correct + unknown_correct) / (len(known_confidences) + len(unknown_confidences)) if (known_confidences or unknown_confidences) else 0
    
    print("\nPerformance with suggested thresholds:")
    print(f"Confidence threshold = {conf_thresh}, Entropy threshold = {entr_thresh}")
    print(f"Known instruments correctly identified: {known_correct}/{len(known_confidences)} ({known_accuracy:.2%})")
    print(f"Unknown sounds correctly marked as Unknown: {unknown_correct}/{len(unknown_confidences)} ({unknown_accuracy:.2%})")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    # Try a few other thresholds and report results
    alternative_thresholds = [
        {'conf': 0.85, 'entr': 0.15},
        {'conf': 0.8, 'entr': 0.15},
        {'conf': 0.9, 'entr': 0.08}
    ]
    
    print("\nPerformance with alternative thresholds:")
    
    for thresh in alternative_thresholds:
        conf_thresh = thresh['conf']
        entr_thresh = thresh['entr']
        
        known_correct = 0
        unknown_correct = 0
        
        for conf, entr in zip(known_confidences, known_entropies):
            is_unknown = conf < conf_thresh or entr > entr_thresh
            if not is_unknown:
                known_correct += 1
        
        for conf, entr in zip(unknown_confidences, unknown_entropies):
            is_unknown = conf < conf_thresh or entr > entr_thresh
            if is_unknown:
                unknown_correct += 1
        
        known_accuracy = known_correct / len(known_confidences) if known_confidences else 0
        unknown_accuracy = unknown_correct / len(unknown_confidences) if unknown_confidences else 0
        overall_accuracy = (known_correct + unknown_correct) / (len(known_confidences) + len(unknown_confidences)) if (known_confidences or unknown_confidences) else 0
        
        print(f"Confidence threshold = {conf_thresh}, Entropy threshold = {entr_thresh}")
        print(f"Known instruments correctly identified: {known_correct}/{len(known_confidences)} ({known_accuracy:.2%})")
        print(f"Unknown sounds correctly marked as Unknown: {unknown_correct}/{len(unknown_confidences)} ({unknown_accuracy:.2%})")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print()
    
    return {
        'confidence_threshold': 0.9, 
        'entropy_threshold': 0.12
    }  # Return the suggested thresholds

def main():
    try:
        print("Loading model and label encoder...")
        # Load the model
        model_path = f"{MODEL_SAVE_PATH}.h5"
        model = tf.keras.models.load_model(model_path)
        
        # Load the label encoder classes
        label_encoder_path = "label_encoder.pkl"
        with open(label_encoder_path, 'rb') as f:
            classes = pickle.load(f)
        
        # Create a simple label encoder using the loaded classes
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = classes
        
        print(f"Model loaded successfully. Classes: {classes}")
        
        # Prepare lists for known and unknown samples
        known_samples = []
        unknown_samples = []
        
        # Find known test samples
        test_dir = "testing_data"  # Update this to your test data path
        if os.path.exists(test_dir):
            known_samples = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.wav')]
            print(f"Found {len(known_samples)} known instrument test files.")
        else:
            print(f"Test directory {test_dir} not found.")
        
        # Find unknown instrument samples
        unknown_dir = "unknown_instruments"  # Your folder with collected non-Lao instruments
        if os.path.exists(unknown_dir):
            unknown_samples = [os.path.join(unknown_dir, f) for f in os.listdir(unknown_dir) if f.endswith('.wav')]
            print(f"Found {len(unknown_samples)} unknown instrument test files.")
        else:
            print(f"Unknown instruments directory {unknown_dir} not found.")
        
        if known_samples and unknown_samples:
            print("\nStarting analysis with manual threshold approach...")
            
            # Step 1: Visualize distributions to understand data
            print("\n1. Visualizing data distributions for manual threshold adjustment...")
            suggested_thresholds = visualize_distributions(model, label_encoder, known_samples, unknown_samples)
            
            # Step 2: Test with manually selected thresholds
            print("\n2. Testing with manually selected thresholds...")
            
            # Manually selected thresholds (Approach 3)
            manual_thresholds = {
                'confidence_threshold': 0.9,
                'entropy_threshold': 0.12
            }
            
            # Run test with manual thresholds
            test_results = test_unknown_detection(model, label_encoder, known_samples, unknown_samples, manual_thresholds)
            
            print("\nTesting complete! You can use these thresholds in your application:")
            print(f"confidence_threshold={manual_thresholds['confidence_threshold']:.2f}, entropy_threshold={manual_thresholds['entropy_threshold']:.2f}")
            
            # Final implementation for production
            print("\nFinal implementation function for your application:")
            print("""
def predict_instrument_final(audio_path, model, label_encoder):
    '''
    Final prediction function with optimal thresholds for production
    '''
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=44100, duration=4)
    
    # Ensure consistent length
    if len(audio) < 4 * 44100:
        audio = np.pad(audio, (0, 4 * 44100 - len(audio)))
    else:
        audio = audio[:4 * 44100]
    
    # Extract features
    mel_spec = extract_features(audio, sr)
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
    
    # Make prediction
    prediction = model.predict(mel_spec, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    # Calculate entropy
    entropy = -np.sum(prediction * np.log2(prediction + 1e-10))
    max_entropy = np.log2(len(prediction))
    normalized_entropy = entropy / max_entropy
    
    # Use manually tuned thresholds
    confidence_threshold = 0.9
    entropy_threshold = 0.12
    
    # Determine if the sound is unknown
    is_unknown = confidence < confidence_threshold or normalized_entropy > entropy_threshold
    
    if is_unknown:
        return "Unknown", confidence, normalized_entropy
    else:
        return label_encoder.classes_[predicted_class], confidence, normalized_entropy
            """)
            
        else:
            print("\nNot enough samples to test. Please ensure both known and unknown sample directories exist.")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()