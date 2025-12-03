import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

def extract_features(audio_file):
    """
    Extract MFCC features from audio file with proper normalization
    Returns: MFCC features array (13 features)
    """
    try:
        # Load audio file with consistent settings
        y, sr = librosa.load(audio_file, sr=22050, duration=2.0)
        
        # Trim silence from beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # If trimming removed too much, use original with padding
        if len(y_trimmed) < sr * 0.3:  # Less than 0.3 seconds of audio
            y_trimmed = y
            print(f"⚠️  Short audio in {os.path.basename(audio_file)}")
        
        # Normalize audio to prevent extreme values
        if np.max(np.abs(y_trimmed)) > 0:
            y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
        else:
            y_normalized = y_trimmed
            print(f"⚠️  Silent audio in {os.path.basename(audio_file)}")
        
        # Extract MFCC features with proper parameters
        mfcc = librosa.feature.mfcc(
            y=y_normalized, 
            sr=sr, 
            n_mfcc=13,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Calculate mean of each coefficient across time
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Debug: Check for extreme values
        if np.max(np.abs(mfcc_mean)) > 100:
            print(f"⚠️  High MFCC values in {os.path.basename(audio_file)}: {mfcc_mean[0]:.2f}, {mfcc_mean[1]:.2f}")
        
        return mfcc_mean
        
    except Exception as e:
        print(f"❌ Error processing {audio_file}: {e}")
        return None

def create_dataset(data_folder):
    """
    Create dataset from all audio files in folder
    Returns: features array, labels array, file_names list
    """
    features = []
    labels = []
    file_names = []
    
    commands = ['shush', 'click', 'whistle', 'pop', 'hiss', 'hum']
    
    for label, command in enumerate(commands):
        command_folder = os.path.join(data_folder, command)
        
        if not os.path.exists(command_folder):
            print(f"⚠️  Folder {command_folder} does not exist")
            continue
            
        print(f"Processing {command}...")
        file_count = 0
        
        for filename in os.listdir(command_folder):
            if filename.endswith('.wav'):
                filepath = os.path.join(command_folder, filename)
                
                # Extract features
                mfcc_features = extract_features(filepath)
                
                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(label)
                    file_names.append(filename)
                    file_count += 1
        
        print(f"  ✅ Processed {file_count} {command} samples")
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"Total samples: {features_array.shape[0]}")
    print(f"Features per sample: {features_array.shape[1]}")
    print(f"MFCC value range: {np.min(features_array):.2f} to {np.max(features_array):.2f}")
    
    return features_array, labels_array, file_names

def plot_features(features, labels):
    """Plot the first two MFCC features to visualize the data"""
    commands = ['shush', 'click', 'whistle', 'pop', 'hiss', 'hum']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(12, 10))
    
    for i, command in enumerate(commands):
        mask = labels == i
        if np.sum(mask) > 0:
            plt.scatter(features[mask, 0], features[mask, 1], 
                       c=colors[i], label=command, alpha=0.7, s=60)
    
    plt.xlabel('MFCC Coefficient 1')
    plt.ylabel('MFCC Coefficient 2')
    plt.title('MFCC Feature Visualization - Fixed Version')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add range information
    x_range = np.max(features[:, 0]) - np.min(features[:, 0])
    y_range = np.max(features[:, 1]) - np.min(features[:, 1])
    plt.figtext(0.5, 0.01, f'MFCC1 Range: {x_range:.2f}, MFCC2 Range: {y_range:.2f}', 
                ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.savefig('mfcc_features_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

def check_feature_quality(features, labels):
    """Check if features are in reasonable ranges"""
    commands = ['shush', 'click', 'whistle', 'pop', 'hiss', 'hum']
    
    print("\n🔍 FEATURE QUALITY CHECK:")
    print("=" * 40)
    
    overall_min = np.min(features)
    overall_max = np.max(features)
    overall_range = overall_max - overall_min
    
    print(f"Overall feature range: {overall_min:.2f} to {overall_max:.2f}")
    print(f"Overall feature span: {overall_range:.2f}")
    
    if overall_range > 1000:
        print("❌ PROBLEM: Features have extreme values!")
        return False
    elif overall_range < 10:
        print("⚠️  WARNING: Features have limited range!")
        return False
    else:
        print("✅ GOOD: Features in reasonable range")
        return True

if __name__ == "__main__":
    # Test the feature extraction
    print("Extracting features from training data...")
    features, labels, filenames = create_dataset("sounds/train")
    
    # Check feature quality
    is_good = check_feature_quality(features, labels)
    
    if is_good:
        print("\n✅ Feature extraction successful! Proceed with training.")
        plot_features(features, labels)
    else:
        print("\n❌ Feature extraction has issues! Check audio files.")