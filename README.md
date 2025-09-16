# Deepfake Audio Detection

This project focuses on detecting synthetic (spoofed) speech using machine learning. 
We use the ASVspoof 2019 Logical Access (LA) dataset to train and evaluate models that classify audio as either bonafide or spoofed.

## Project Overview
The goal of this project is to automatically detect deepfake audio samples. 
We preprocess the audio, extract features like MFCC and CQCC, train a convolutional neural network (CNN), and evaluate performance using metrics such as EER and t-DCF.

## Technologies / Libraries Used
- **Python 3.10+** – main programming language
- **PyTorch** – deep learning framework
- **torchaudio** – audio processing for PyTorch models
- **librosa** – audio feature extraction (MFCC, CQCC)
- **NumPy** – numerical computing and array manipulation
- **pandas** – handling protocol files and tabular data
- **scikit-learn** – evaluation metrics and utilities
- **Matplotlib / Seaborn** – plotting and visualizations
- **HuggingFace Transformers** – optional, for experimenting with pretrained models
- **Dataset:** ASVspoof 2019 LA (Logical Access)
- **File formats:** WAV audio, NumPy `.npy` for preprocessed features

## Dataset
We use the ASVspoof 2019 LA dataset:
- 3 subsets: Train, Development, Evaluation
- Audio format: 16 kHz, mono, 16-bit WAV
- Protocol files indicate bonafide vs spoofed samples
- Dataset link: [ASVspoof 2019](https://www.asvspoof.org)

## Data Preprocessing
- Read protocol files to identify audio type
- Extract features: MFCC initially, later CQCC
- Handle missing or corrupted files gracefully
- Save processed features and labels as `.npy` files for efficient training

## Model
- Baseline model: CNN with residual blocks
- Input: MFCC or CQCC features
- Output: Probability of bonafide or spoofed
- Training: PyTorch, torchaudio, librosa

## Evaluation
We evaluate models using:
- **EER** (Equal Error Rate)
- **t-DCF** (tandem Detection Cost Function)

These metrics are standard for spoofed audio detection and allow comparison with published work.

## Results / Achievements
- Successfully preprocessed ASVspoof 2019 LA dataset and extracted MFCC/CQCC features
- Baseline CNN model achieved ~90% test accuracy using CQCC features
- Fully saved processed features and labels for Train, Dev, and Eval sets in `.npy` format
- Ready for further model training, hyperparameter tuning, and evaluation

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Murad2111/DeepfakeAudioDetection.git


