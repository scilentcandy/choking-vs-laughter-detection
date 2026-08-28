# Choking vs. Laughter Detection Using Audio Classification

This repository contains the code developed for a final project on automatic audio classification of **choking** and **laughter** sounds.

The project investigates whether machine-learning models can distinguish these acoustically similar sound classes using a multi-feature audio representation and several neural-network architectures.

## Project pipeline

The code is organized around the following workflow:

1. **Raw-audio preparation**
   - resampling and cleanup
   - noise-floor handling
   - event-aware windowing
   - silence/content screening
   - normalization

2. **Audio preprocessing and feature extraction**
   - source-level train/validation/test splitting
   - raw-signal augmentation
   - pre-emphasis
   - extraction of a 20-channel feature representation
   - train-set normalization statistics

3. **Model training and evaluation**
   - Feed-Forward Neural Network (FNN)
   - Flat Convolutional Neural Network (CNN)
   - Multi-Fusion CNN
   - Recurrent Neural Network (BiLSTM/RNN)
   - Convolutional Recurrent Neural Network (CRNN)

4. **Robustness and feature analysis**
   - six-run architecture comparison
   - RNN feature-group importance analysis
   - Multi-Fusion CNN branch-importance analysis

## Repository structure

```text
choking-vs-laughter-detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── preprocessing/
│   ├── prepare_raw.py
│   └── audio_preprocessing_v3.py
│
├── notebooks/
│   ├── fnn.ipynb
│   ├── flat_cnn.ipynb
│   ├── multi_fusion_cnn.ipynb
│   ├── rnn.ipynb
│   └── crnn.ipynb
│
└── experiments/
    ├── rnn_feature_importance.ipynb
    ├── multi_fusion_cnn_feature_importance.ipynb
    └── architecture_comparison_6_runs.ipynb
```

## Audio representation

The preprocessing pipeline extracts a 20-channel representation containing spectral, temporal, tonal, voicing, and energy-related information. The active channels in the uploaded preprocessing code are:

- Log-Mel spectrogram
- Harmonic Mel spectrogram
- Percussive Mel spectrogram
- Delta
- Delta-Delta
- Zero-Crossing Rate (ZCR)
- RMS energy
- Spectral centroid
- Spectral rolloff
- Spectral bandwidth
- Spectral contrast
- Chroma
- MFCC
- Tonnetz
- Spectral flatness
- Fundamental frequency (F0)
- Voiced probability
- Spectral flux
- Wide-band Mel spectrogram
- RMS × periodicity

## Data preparation

The raw preparation stage expects audio under:

```text
data/raw/choking/
data/raw/laughter/
```

Running `preprocessing/prepare_raw.py` creates fixed-length prepared windows under `data/raw_prepared/` and records preparation metadata.

The feature-extraction stage then processes the prepared audio into train, validation, and test tensors. Source recordings are grouped before splitting so that windows originating from the same recording remain in the same split.

The generated audio, tensors, metadata, checkpoints, and other large experiment artifacts are intentionally excluded from this repository.

## Running the notebooks

The training notebooks were developed in **Google Colab** and retain their original experiment outputs, including logs, plots, and evaluation results.

Several notebooks contain a `PROJECT_DIR` pointing to the original Google Drive project directory. Change that variable to match your own Drive or local project location before running the notebooks.

Some notebooks correspond to different stages of the project and therefore preserve their experiment-specific processed-data paths and settings.

## Installation

For the Python preprocessing scripts, install the dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch training is generally easiest to run in Google Colab with a GPU runtime.

## Dataset

The audio dataset is **not included** in this repository. Raw recordings, prepared audio, processed NumPy tensors, and trained model checkpoints are excluded to keep the repository lightweight and to avoid distributing project data unintentionally.

## Authors

Authors will be added later.
