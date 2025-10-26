# 🗣️ Non-Linguistic Vocal Command Recognition System


## 📘 Introduction

**Non-Linguistic Vocal Command Recognition (NLVCR)** offers a new approach for human-computer interaction by using non-speech sounds, such as **shush**, **click**, **whistle**, **pop**, **hiss**, and **hum**, as control signals for devices. This system is also integrated with a Music Player for real-world application.

These cues are:
- **Language-independent**
- **Discreet and private**
- **Less affected by background noise**
- **Ideal for hands-free interaction**

This project applies **signal processing** and **machine learning** techniques to recognize and classify these sounds for easy control in smart environments.


## 🧠 System Architecture

The system consists of the following major components:

1. **Data Collection**
2. **Feature Extraction (MFCC)**
3. **Model Training and Evaluation**
4. **Real-Time Command Recognition**
5. **Integration with Music Player**


## 🎙️ Data Collection

- **Commands:** `shush`, `click`, `whistle`, `pop`, `hiss`, `hum`
- **Recording Library:** `sounddevice`
- **Sampling Rate:** `22.05 kHz`
- **Recording Duration:** `2 seconds per sample`
- **Storage Format:** `.wav`

Each command is recorded multiple times and organized into separate training and testing directories.

Example filenames: `shush_1.wav`, `click_test_2.wav`


Interactive data collection ensures consistent amplitude normalization and systematic file naming.


## 🔍 Feature Extraction

- **Method:** Mel-Frequency Cepstral Coefficients (**MFCCs**)
- **Number of Coefficients:** 13
- **Libraries Used:** `librosa`, `scipy`, `matplotlib`
- **Purpose:** Convert complex acoustic signals into fixed-length numerical feature vectors.

### Visualization

- The extracted features are plotted on a **2D scatter plot** (`mfcc_features_plot.png`).
- Helps in observing the separability of different vocal commands.


## 🤖 Model Training

- **Model:** Support Vector Machine (**SVM**) with Linear Kernel
- **Frameworks:** `scikit-learn`, `joblib`
- **Dataset Split:** 75% Training, 25% Validation (Stratified)
- **Performance Metrics:** Training and Test Accuracy, Confusion Matrix

The Test and Training Accuracy for various samples are shown in the table below - 

| Dataset Size | Training Accuracy | Test Accuracy |
|---------------|------------------|----------------|
| 30 Samples | 88% | 39% |
| 120 Samples | 60% | 57% |
| 180 Samples | 80% | 72% |

Model saved as: `vocal_command_model.pkl`


## 🧩 Command Recognition

- **Input:** Real-time audio (2-second recording)
- **Processing:** MFCC extraction (13 features)
- **Classification:** Pre-trained SVM model
- **Output:** Predicted command + confidence probabilities

### Supported Commands

| Command | Action |
|----------|---------|
| `shush` | Pause Music |
| `click` | Resume Music |
| `whistle` | Skip Track |
| `pop` | Previous Track |
| `hiss` | Volume Down |
| `hum` | Volume Up |

### Modes

- **Single Prediction Mode:** Records and predicts one command per input.
- **Continuous Listening Mode:** Listens continuously until manually stopped.


## 📊 Results & Visualization

- MFCC Feature plots show clear class clusters.
- Confusion matrices visualize classification performance and misclassifications.

## ⚙️ Execution Order

Follow this sequence to run the project properly:

```bash
# 1. Collect data
python data_collection_fixed.py

# 2. Check the data
python data_cleanup.py

# 3. Extract features
python feature_extraction.py

# 4. Train model  
python model_training.py

# 5. Run the music player with built-in voice control
python music_player.py
