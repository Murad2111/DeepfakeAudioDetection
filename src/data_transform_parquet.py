import gc

import pandas as pd
import os
import librosa
import io
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.fftpack import dct
import warnings
from pandarallel import pandarallel

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


def audio_to_cqcc(audio_dict, n_coeffiecients=20, hop_length=128, fmin=20, n_bins=96, bins_per_octave=12):
    """
    Convert audio to CQCC spectrogram.
    Returns: CQCC array [n_coeffiecients x time]
    """
    import io, warnings, librosa
    import numpy as np
    import soundfile as sf
    audio_bytes = io.BytesIO(audio_dict["bytes"])
    y, sr = sf.read(audio_bytes, dtype='float32')
    # If stereo, take mean to make it mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    warnings.filterwarnings("ignore", message="n_fft=.* is too large")
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
    #return C   # if we want to visualize it
    log_C = np.log(C + 1e-8)        #for numerical stuff so that all the values are not too far apart (distance is kept)
    cqcc = dct(log_C, type=2, axis=0, norm='ortho')
    cqcc = cqcc[:n_coeffiecients, :]  #only keep first n coefficients
    return cqcc


def visualize_spectrogram(spectrogram):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, x_axis='time')
    plt.colorbar()
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()


def convert_dataset(conversion_function):
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

    #tqdm.pandas()
    pandarallel.initialize(progress_bar=True, nb_workers=8)
    ret_test_mfcc = df_test['audio'].parallel_apply(conversion_function)
    gc.collect()
    ret_train_mfcc = df_train['audio'].parallel_apply(conversion_function)
    gc.collect()
    ret_val_mfcc = df_val['audio'].parallel_apply(conversion_function)
    gc.collect()

    return  ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc, ret_val_mfcc


def safe_dataset(test_label, train_label, val_label, test_mfcc, train_mfcc, val_mfcc, path):
    """
    safes the given DataFrames as 3 parquet files
    """
    np.save(path+r"\test_labels.npy", test_label)
    np.save(path + r"\train_labels.npy", train_label)
    np.save(path + r"\val_labels.npy", val_label)

    np.save(path + r"\test_features.npy", test_mfcc)
    np.save(path + r"\train_features.npy", train_mfcc)
    np.save(path + r"\val_features.npy", val_mfcc)


def do_files_exist(path):
    """
    checks if the processed dataset exists
    :return: true if dataset exists, false if not
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)

    return os.path.isfile(path+ r"\val_features.npy")


def check_processed_datasets(path, conversion_function):
    """
    makes sure all necessary datasets exist
    """
    if not do_files_exist(path):
        print("processed datasets does not exist yet")
        print("-----------------converting datasets-------------------------")
        (ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc,
         ret_val_mfcc) = convert_dataset(conversion_function)
        safe_dataset(ret_test_label, ret_train_label, ret_val_label, ret_test_mfcc, ret_train_mfcc, ret_val_mfcc, path)
        print("--------------finished converting datasets------------------")
    else:
        print("found processed datasets")


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
    #df['mfcc'] = df['audio'].progress_apply(audio_to_cqcc)
    #visualize_spectrogram(df["mfcc"].iloc[0])
    visualize_spectrogram(audio_to_cqcc(df['audio'].iloc[0], n_coeffiecients=20))


if __name__ == "__main__":
    #do_some_tests()
    path = os.getcwd() + r"\..\data\processed\CQCC2"
    check_processed_datasets(path, audio_to_cqcc)