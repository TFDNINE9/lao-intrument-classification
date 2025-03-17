import numpy as np
import librosa
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt

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
                                             confidence_threshold=0.7, entropy_threshold=0.6):
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
        is_unknown = (confidence < confidence_threshold or 
                     normalized_entropy > entropy_threshold)
        
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

def generate_test_noise_samples(num_samples=5, save_dir="test_noise_samples"):
    """
    Generate synthetic noise samples for testing unknown detection
    
    Parameters:
    num_samples: Number of each type of noise to generate
    save_dir: Directory to save generated samples
    
    Returns:
    list: Paths to generated samples
    """
    import soundfile as sf
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    sample_paths = []
    
    # Generate samples
    for i in range(num_samples):
        # White noise
        white_noise = np.random.normal(0, 0.1, SAMPLE_RATE * DURATION)
        white_noise_path = os.path.join(save_dir, f"white_noise_{i}.wav")
        sf.write(white_noise_path, white_noise, SAMPLE_RATE)
        sample_paths.append(white_noise_path)
        
        # Pink noise (more bass)
        pink_noise = np.random.normal(0, 0.1, SAMPLE_RATE * DURATION)
        # Apply filtering to make it pink
        pink_noise = np.cumsum(pink_noise)
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.5
        pink_noise_path = os.path.join(save_dir, f"pink_noise_{i}.wav")
        sf.write(pink_noise_path, pink_noise, SAMPLE_RATE)
        sample_paths.append(pink_noise_path)
        
        # Sine wave (pure tone)
        t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz = A4 note
        sine_wave_path = os.path.join(save_dir, f"sine_wave_{i}.wav")
        sf.write(sine_wave_path, sine_wave, SAMPLE_RATE)
        sample_paths.append(sine_wave_path)
        
        # Chirp
        chirp = np.linspace(0, 1, SAMPLE_RATE * DURATION)
        chirp = 0.5 * np.sin(2 * np.pi * 100 * chirp * chirp)  # Increasing frequency
        chirp_path = os.path.join(save_dir, f"chirp_{i}.wav")
        sf.write(chirp_path, chirp, SAMPLE_RATE)
        sample_paths.append(chirp_path)
    
    return sample_paths
def test_unknown_detection(model, label_encoder, known_samples, threshold_config=None):
    """
    Test unknown detection system and visualize results
    
    Parameters:
    model: Trained model
    label_encoder: Label encoder
    known_samples: List of paths to known instrument samples
    threshold_config: Dictionary with confidence_threshold and entropy_threshold
    
    Returns:
    dict: Stats about detection performance
    """
    if threshold_config is None:
        threshold_config = {
            'confidence_threshold': 0.7,
            'entropy_threshold': 0.6
        }
    
    # Generate unknown samples for testing
    unknown_samples = generate_test_noise_samples(num_samples=3)
    
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
    print("\nTesting with unknown/noise samples:")
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

def adjust_thresholds_interactive(model, label_encoder, known_samples, initial_config=None):
    """
    Interactive tool to adjust thresholds for unknown detection
    
    Parameters:
    model: Trained model
    label_encoder: Label encoder
    known_samples: List of paths to known instrument samples
    initial_config: Dictionary with initial confidence_threshold and entropy_threshold
    
    Returns:
    dict: Final threshold configuration
    """
    if initial_config is None:
        threshold_config = {
            'confidence_threshold': 0.7,
            'entropy_threshold': 0.6
        }
    else:
        threshold_config = initial_config.copy()
    
    # Generate unknown samples for testing
    unknown_samples = generate_test_noise_samples(num_samples=3)
    
    # Collect initial data
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
        except Exception as e:
            print(f"Error processing {sample_path}: {str(e)}")
    
    # Process unknown samples
    print("Analyzing unknown/noise samples...")
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
    
    # Interactive adjustment
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    if len(known_confidences) > 0 and len(unknown_confidences) > 0:
        def update_plot(confidence_threshold, entropy_threshold):
            clear_output(wait=True)
            
            plt.figure(figsize=(12, 5))
            
            # Confidence plot
            plt.subplot(1, 2, 1)
            plt.hist(known_confidences, alpha=0.5, label='Known Instruments')
            plt.hist(unknown_confidences, alpha=0.5, label='Unknown Sounds')
            plt.axvline(x=confidence_threshold, color='r', linestyle='--', label=f'Threshold ({confidence_threshold:.2f})')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title('Confidence Distribution')
            plt.legend()
            
            # Entropy plot
            plt.subplot(1, 2, 2)
            plt.hist(known_entropies, alpha=0.5, label='Known Instruments')
            plt.hist(unknown_entropies, alpha=0.5, label='Unknown Sounds')
            plt.axvline(x=entropy_threshold, color='r', linestyle='--', label=f'Threshold ({entropy_threshold:.2f})')
            plt.xlabel('Normalized Entropy')
            plt.ylabel('Count')
            plt.title('Entropy Distribution')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Calculate performance metrics
            known_correct = sum(1 for c, e in zip(known_confidences, known_entropies) 
                               if c >= confidence_threshold and e <= entropy_threshold)
            known_total = len(known_confidences)
            
            unknown_correct = sum(1 for c, e in zip(unknown_confidences, unknown_entropies)
                                 if c < confidence_threshold or e > entropy_threshold)
            unknown_total = len(unknown_confidences)
            
            known_accuracy = known_correct / known_total if known_total > 0 else 0
            unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0
            overall_accuracy = (known_correct + unknown_correct) / (known_total + unknown_total)
            
            print(f"Confidence threshold: {confidence_threshold:.2f}")
            print(f"Entropy threshold: {entropy_threshold:.2f}")
            print("\nPerformance with current thresholds:")
            print(f"Known instruments correctly identified: {known_correct}/{known_total} ({known_accuracy:.2%})")
            print(f"Unknown sounds correctly marked as Unknown: {unknown_correct}/{unknown_total} ({unknown_accuracy:.2%})")
            print(f"Overall accuracy: {overall_accuracy:.2%}")
        
        # Create interactive sliders
        confidence_slider = widgets.FloatSlider(
            value=threshold_config['confidence_threshold'],
            min=0.0, max=1.0, step=0.05,
            description='Confidence:'
        )
        
        entropy_slider = widgets.FloatSlider(
            value=threshold_config['entropy_threshold'],
            min=0.0, max=1.0, step=0.05,
            description='Entropy:'
        )
        
        # Link sliders to plot update
        widgets.interactive(update_plot, 
                           confidence_threshold=confidence_slider,
                           entropy_threshold=entropy_slider)
        
        # Initial plot
        update_plot(threshold_config['confidence_threshold'], threshold_config['entropy_threshold'])
        
        # Return final thresholds
        threshold_config['confidence_threshold'] = confidence_slider.value
        threshold_config['entropy_threshold'] = entropy_slider.value
    
    return threshold_config

# Main function to load model and test with unknown detection
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
        
        # Set default thresholds
        threshold_config = {
            'confidence_threshold': 0.7,
            'entropy_threshold': 0.6
        }
        
        # Test with existing test files
        test_dir = "testing_data"  # Update this to your test data path
        if os.path.exists(test_dir):
            test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.wav')]
            
            if test_files:
                print(f"\nFound {len(test_files)} test files.")
                
                # Test unknown detection performance
                test_unknown_detection(model, label_encoder, test_files, threshold_config)
                
                # Optional: Adjust thresholds interactively (requires IPython/Jupyter)
                try:
                    import ipywidgets
                    adjusted_config = adjust_thresholds_interactive(model, label_encoder, test_files, threshold_config)
                    print(f"\nAdjusted thresholds: Confidence = {adjusted_config['confidence_threshold']:.2f}, Entropy = {adjusted_config['entropy_threshold']:.2f}")
                except ImportError:
                    print("\nInteractive threshold adjustment requires ipywidgets. Skipping.")
            else:
                print(f"No test files found in {test_dir}")
        else:
            print(f"Test directory {test_dir} not found.")
            
            # Generate and test with synthetic samples only
            noise_samples = generate_test_noise_samples(num_samples=5)
            print(f"Generated {len(noise_samples)} synthetic test samples.")
            
            # Test the synthetic samples
            for sample_path in noise_samples:
                result = predict_instrument_with_unknown_detection(
                    sample_path, model, label_encoder,
                    confidence_threshold=threshold_config['confidence_threshold'],
                    entropy_threshold=threshold_config['entropy_threshold']
                )
                
                if result:
                    print(f"File: {os.path.basename(sample_path)}")
                    print(f"Predicted: {result['instrument']} (Confidence: {result['confidence']:.2f}, Entropy: {result['entropy']:.2f})")
                    print(f"All probabilities: {result['all_probabilities']}")
                    print()
            
        print("\nTesting complete! You can now use predict_instrument_with_unknown_detection function in your application.")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()