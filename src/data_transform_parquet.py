import pandas as pd
import os
import librosa
import io
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

#pyarrow needs to be installed



def audio_to_mfcc(audio_dict, n_mfcc=13, sr=16000):
    """
    Convert audio to MFCC spectrogram.
    Returns: MFCC array [n_mfcc x time]
    """
    audio_bytes = io.BytesIO(audio_dict["bytes"])
    y, sr = sf.read(audio_bytes, dtype='float32')
    # If stereo, take mean to make it mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def visualize_mfcc(mfcc):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def convert_dataset():
    """
    converts the whole dataset to mfcc
    :return: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    """
    path = os.getcwd() + r"\.."
    test = path + r"\data\raw\test-00000-of-00001.parquet"
    train = path + r"\data\raw\train-00000-of-00001.parquet"
    val = path + r"\data\raw\validation-00000-of-00001.parquet"

    df_test = pd.read_parquet(test)
    df_train = pd.read_parquet(train)
    df_val = pd.read_parquet(val)

    ret_test_label = df_test["key"]
    ret_train_label = df_train["key"]
    ret_val_label = df_val["key"]

    tqdm.pandas()
    ret_test_mfcc = df_test['audio'].progress_apply(audio_to_mfcc)
    ret_train_mfcc = df_train['audio'].progress_apply(audio_to_mfcc)
    ret_val_mfcc = df_val['audio'].progress_apply(audio_to_mfcc)

    return  ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc, ret_val_mfcc


def safe_dataset(test_label, train_label, val_label, test_mfcc, train_mfcc, val_mfcc):
    """
    safes the given DataFrames as 3 parquet files
    """
    path = os.getcwd() + r"\..\data\processed\parquet"

    np.save(path+r"\test_labels.npy", test_label)
    np.save(path + r"\train_labels.npy", train_label)
    np.save(path + r"\val_labels.npy", val_label)

    np.save(path + r"\test_features.npy", test_mfcc)
    np.save(path + r"\train_features.npy", train_mfcc)
    np.save(path + r"\val_features.npy", val_mfcc)


def check_processed_dataset():
    """
    checks if the processed dataset exists
    :return: true if dataset exists, false if not
    """
    path = os.getcwd() + r"\..\data\processed\parquet"
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)

    return os.path.isfile(path+ r"\val_features.npy")


def do_some_tests():
    path = os.getcwd() + r"\.."
    filepath = path + r"\data\raw\test-00000-of-00001.parquet"
    df = pd.read_parquet(filepath)
    print("-----------------------data-------------------")
    print("shape: " + str(df.shape))
    print("collumns: " + str(df.columns))
    print("data types: " + str(df.dtypes))
    print("----------info---------------\n" + str(df.info))
    print("----------stats--------------\n"+ str(df.describe))


    audioData = df["audio"]
    label = df["key"]

    print("------------------audio and keys-------------------")
    print(label[0])
    print(audioData[0].keys())

    tqdm.pandas()
    df['mfcc'] = df['audio'].progress_apply(audio_to_mfcc)
    visualize_mfcc(df["mfcc"].iloc[0])


if __name__ == "__main__":
    #do_some_tests()
    if not check_processed_dataset():
        ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc, ret_val_mfcc = convert_dataset()
        safe_dataset(ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc, ret_val_mfcc)