# -------------------------------
# Imports
# -------------------------------
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
# -------------------------------
# Define dataset paths
# -------------------------------
path = os.getcwd() + r"\.."
datasets = {
    "train": {
        "audio_path": path + r"\data\raw\LA\train",
        "protocol_path": path + r"\data\raw\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt",
        "features_file": path + r"\data\processed\train_features.npy",
        "labels_file": path + r"\data\processed\train_labels.npy"
    },
    "dev": {
        "audio_path": path + r"\data\raw\LA\dev",
        "protocol_path": path + r"\data\raw\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt",
        "features_file": path + r"\data\processed\dev_features.npy",
        "labels_file": path + r"\data\processed\dev_labels.npy"
    },
    "eval": {
        "audio_path": path + r"\data\raw\LA\eval",
        "protocol_path": path + r"\data\raw\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt",
        "features_file": path + r"\data\processed\eval_features.npy",
        "labels_file": path + r"\data\processed\eval_labels.npy"
    }
}

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features(audio_file_path, sr=16000, n_mfcc=13):
    try:
        y, _ = librosa.load(audio_file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_dataset(dataset_name, audio_path, protocol_path, features_file, labels_file):
    print(f"\nProcessing {dataset_name} dataset ...")

    # Load protocol
    protocol_df = pd.read_csv(
        protocol_path,
        sep=r'\s+',  # raw string for whitespace splitting
        header=None,
        names=["file_id", "audio_file", "speaker", "system"]
    )

    features = []
    labels = []

    # Iterate over all entries
    for idx, row in tqdm(protocol_df.iterrows(), total=len(protocol_df)):
        file_path = os.path.join(audio_path, row["audio_file"] + ".wav")
        if os.path.exists(file_path):
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(row["system"])
        else:
            print(f"Warning: file not found {file_path}")

    features = np.array(features)
    labels = np.array(labels)

    os.makedirs(os.path.dirname(features_file), exist_ok=True)

    # Save features and labels
    np.save(features_file, features)
    np.save(labels_file, labels)

    print(f"{dataset_name} dataset preprocessing complete! Features and labels saved.")

# -------------------------------
# Run preprocessing for all datasets
# -------------------------------
for name, paths in datasets.items():
    preprocess_dataset(
        name,
        paths["audio_path"],
        paths["protocol_path"],
        paths["features_file"],
        paths["labels_file"]
    )