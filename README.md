# Training a Neural Network to Understand Spoken Commands

## Group Members
- Nicholas Hefner (014501147)
- Arthur Ho (025847586)
- Hsuan-Yu Lin (035276148)

## Dataset
**Link:** [TensorFlow Speech Commands Dataset](https://www.tensorflow.org/datasets/catalog/speech_commands)

The dataset consists of over 105,000 audio files in WAV format of people saying 35 different words.

## Project Description
The purpose of this dataset provided by Google's Brain Team is to allow developers to use speech to control their application. In this dataset, there are 65,000 audio files of 30 different spoken commands, such as "Yes", "No", "Stop", and even spoken names. This dataset includes background noise to allow for data augmentation for more robust training data.

The goal of this project is to train and test different convolutional neural networks with varying depth and regularization strategies to compare their accuracy on the Speech Commands dataset. This comparison will help determine how model complexity interacts with dataset variability in influencing recognition performance.

## Project Structure
```
CECS_550_Speech_Commands_CNN/
├── baseline_cnn_model.ipynb      # Baseline CNN model using spectrograms
├── traditional_ml_mfcc.ipynb     # Traditional ML models using MFCC features
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/                         # Dataset (auto-downloaded)
├── models/                       # Saved models
└── figures/                      # Generated plots and visualizations
```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   - Start with `baseline_cnn_model.ipynb` to train the baseline CNN
   - Then run `traditional_ml_mfcc.ipynb` to compare with traditional ML models

## Notebooks

### 1. Baseline CNN Model (`baseline_cnn_model.ipynb`)
Implements a CNN model following the [TensorFlow Simple Audio Recognition Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio):
- Uses spectrogram (STFT) features
- 2 Conv2D layers + Dense layers
- Includes training, evaluation, and visualization

### 2. Traditional ML with MFCC (`traditional_ml_mfcc.ipynb`)
Compares traditional ML approaches using MFCC features:
- Extracts MFCC + delta + delta-delta features using librosa
- Tests SVM, Random Forest, and KNN classifiers
- Provides comparison metrics

## References
- [TensorFlow Simple Audio Recognition Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Speech Commands Dataset Paper (Warden, 2018)](https://arxiv.org/abs/1804.03209)
