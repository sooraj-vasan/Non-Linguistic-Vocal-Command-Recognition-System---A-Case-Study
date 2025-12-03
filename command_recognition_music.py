import sounddevice as sd
import numpy as np
import librosa
import joblib
import time
import os

class VocalCommandRecognizer:
    def __init__(self, model_path):
        """Initialize the recognizer with a trained model"""
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found!")
            print("Please run model_training.py first to train the model.")
            return
        
        self.model = joblib.load(model_path)
        # Updated commands list with music-specific actions
        self.commands = ['shush', 'click', 'whistle', 'pop', 'hiss', 'hum']
        self.sample_rate = 22050
        self.duration = 2  # seconds
        self.chunk_size = 1024
        
        print(f"🎵 MUSIC CONTROL - VOCAL COMMAND RECOGNITION")
        print("=" * 50)
        print("Loaded model for commands:")
        for i, cmd in enumerate(self.commands):
            print(f"  {i}: {cmd}")
        print("\nMusic Control Actions:")
        print("  shush   : ⏸️  Pause music")
        print("  click   : ▶️  Play/Resume music") 
        print("  whistle : ⏭️  Next track")
        print("  pop     : 🔊 Volume up")
        print("  hiss    : 🔈 Volume down")
        print("  hum     : ⏮️  Previous track")
        print("=" * 50)
    
    def record_audio(self):
        """Record audio with real-time feedback"""
        print(f"\nRecording for {self.duration} seconds...", end='', flush=True)
        
        audio = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio error: {status}")
            audio.append(indata.copy())
        
        with sd.InputStream(samplerate=self.sample_rate, 
                          channels=1, 
                          blocksize=self.chunk_size,
                          callback=callback):
            sd.sleep(int(self.duration * 1000))
        
        # Combine all chunks
        audio = np.concatenate(audio, axis=0).flatten()
        print(" Done!")
        return audio
    
    def extract_features(self, audio):
        """Extract MFCC features from audio data"""
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract MFCC features (same as training)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        return mfcc_mean.reshape(1, -1)
    
    def predict_command(self, features):
        """Predict command from features"""
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return prediction, confidence, probabilities
    
    def execute_music_command(self, command, confidence, music_player):
        """Execute music control action based on detected command"""
        actions = {
            'shush': "⏸️  Pausing music",
            'click': "▶️  Playing music", 
            'whistle': "⏭️  Skipping to next track",
            'pop': "🔊 Volume up",
            'hiss': "🔈 Volume down",
            'hum': "⏮️  Previous track"
        }
        
        action_text = actions.get(command, f"❓ Unknown command: {command}")
        print(f"🎯 {action_text} (confidence: {confidence:.2f})")
        
        # Execute the actual music control
        if command == 'shush':
            music_player.pause()
        elif command == 'click':
            music_player.play()
        elif command == 'whistle':
            music_player.next_track()
        elif command == 'pop':
            music_player.volume_up()
        elif command == 'hiss':
            music_player.volume_down()
        elif command == 'hum':
            music_player.previous_track()