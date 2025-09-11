import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import data_transform_parquet
import math


class MyDataset(Dataset): #because our features have variate sequence length
    #maybe you have a better solution
    def __init__(self, X_iterable, y_iterable):
        self.X = X_iterable           # list of np arrays
        self.y = y_iterable

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.from_numpy(self.X[idx]).float()
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_tensor, y_tensor


#------------------------------utility functions----------------------------------------
# Custom collate function
def _pad_collate(batch):
    Xs, ys = zip(*batch)

    #max_len = max(x.shape[1] for x in Xs)  #replace with largest across whole dataset maybe if conv dimensions have trouble
    max_len = 414 #longest sequence in our dataset +1

    # Pad all to max_len along time axis
    Xs_padded = [] #paddings for all x in batch
    for x in Xs:
        pad_size = max_len - x.shape[1]
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size))  #zero pad
        Xs_padded.append(x)

    Xs_padded = torch.stack(Xs_padded).unsqueeze(1)   #get tensors
    # unsqueze to get right dimensions (batch, x, y) -> (batch, 1, x, y)
    ys = torch.stack(ys).unsqueeze(1)

    return Xs_padded, ys


def create_data_loaders(path, batch_size=16):
    """
    creates dataloaders from the processed npy files
    :param path: path to files (str)
    :param batch_size: size of the batches :) int
    :return: test_dataloader, train_dataloader, val_dataloader all are torch.utils.data.DataLoader
    """
    data_transform_parquet.check_processed_datasets(path)

    test_features = np.load(path + r"\test_features.npy", allow_pickle=True)
    test_labels= np.load(path + r"\test_labels.npy", allow_pickle=True)
    train_features = np.load(path + r"\train_features.npy", allow_pickle=True)
    train_labels = np.load(path + r"\train_labels.npy", allow_pickle=True)
    val_features = np.load(path + r"\val_features.npy", allow_pickle=True)
    val_labels = np.load(path + r"\val_labels.npy", allow_pickle=True)

    #test_dataset = TensorDataset(test_feature_tensors, test_label_tensors)
    #train_dataset = TensorDataset(train_feature_tensors, train_label_tensors)
    #val_dataset = TensorDataset(val_feature_tensors, val_label_tensors)

    test_dataset = MyDataset(test_features, test_labels)
    train_dataset = MyDataset(train_features, train_labels)
    val_dataset = MyDataset(val_features, val_labels)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=_pad_collate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=_pad_collate)

    return test_dataloader, train_dataloader, val_dataloader


def _find_largest_sequence(test_loader, train_loader, val_loader):
    max = 0
    for loader in (test_loader, train_loader, val_loader):
        for i, (x, y) in enumerate(loader):
            cur_longest = np.shape(x)[-1]
            if max < cur_longest:
                max = cur_longest
    return max