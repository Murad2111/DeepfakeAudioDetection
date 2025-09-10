import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
import torch
import numpy as np
import os
import load_data

#-----------------------------classes--------------------------------------------------------------
class ConvNet(torch.nn.Module):
    """
    change this later
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=4)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(1616, 120) #out_channels*x_cur*y_cur
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 1)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool(x)

        x = F.sigmoid(self.conv2(x))
        x = self.pool(x)

        # flatten the convolution results
        x = x.view(x.size(0), -1)  #out_channels*x_cur*y_cur

        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return self.output(x)


#------------------------------utility functions----------------------------------------
def train(network, train_loader, val_loader, test_loader, device, epochs=200, lr=1e-3):
    """
    copied from exercise for now
    """
    loss_fn = torch.nn.BCEWithLogitsLoss() #our network should output a single probability imo
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    num_train_samples = (len(train_loader) * train_loader.batch_size)
    num_val_samples = (len(val_loader) * val_loader.batch_size)
    num_test_samples = (len(test_loader) * test_loader.batch_size)

    best_model_valid_loss = np.Inf
    num_epochs_without_val_loss_reduction = 0
    early_stopping_window = 5

    for epoch in range(epochs):
        running_train_loss = 0
        running_val_loss = 0

        correct_train_preds = 0
        correct_val_preds = 0

        # iterate the training data
        with tqdm(train_loader, desc="Training") as train_epoch_pbar:
            for i, (x, y) in enumerate(train_epoch_pbar):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = network(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                running_train_loss += loss * len(x)
                correct_train_preds += (logits.argmax(-1) == y).sum()

                if i % 10 == 0:
                    # we can divide by (i * len(x)) because we are dropping the last batch
                    num_train_samples_so_far = (i + 1) * len(x)
                    train_epoch_pbar.set_postfix(train_loss=running_train_loss.item() / num_train_samples_so_far,
                                                 accuracy=correct_train_preds.item() / num_train_samples_so_far * 100)

        # iterate the val data
        with tqdm(val_loader, desc="Validating") as val_epoch_pbar:
            for i, (x, y) in enumerate(val_epoch_pbar):
                x, y = x.to(device), y.to(device)
                logits = network(x)
                loss = loss_fn(logits, y)

                running_val_loss += loss * len(x)
                correct_val_preds += (logits.argmax(-1) == y).sum()

                if i % 10 == 0:
                    num_val_samples_so_far = (i + 1) * len(x)
                    val_epoch_pbar.set_postfix(train_loss=running_val_loss.item() / num_val_samples_so_far,
                                               accuracy=correct_val_preds.item() / num_val_samples_so_far * 100)

        avg_train_loss = running_train_loss.item() / num_train_samples
        train_acc = correct_train_preds.item() / num_train_samples * 100
        avg_val_loss = running_val_loss.item() / num_val_samples
        val_acc = correct_val_preds.item() / num_val_samples * 100

        print(
            f'Epoch {epoch}: \tAvg Train Loss: {avg_train_loss:.2f} \tTrain Acc: {train_acc:.2f} \tAvg Val Loss: {avg_val_loss:.2f} \tVal Acc: {val_acc:.2f}')

        # perform early stopping if necessary
        if avg_val_loss <= best_model_valid_loss:
            print(f'Validation loss decreased ({best_model_valid_loss:.6f} --> {avg_val_loss:.6f}).  Saving model ...')
            torch.save(network.state_dict(), 'model.pt')
            best_model_valid_loss = avg_val_loss
            num_epochs_without_val_loss_reduction = 0
        else:
            num_epochs_without_val_loss_reduction += 1

        if num_epochs_without_val_loss_reduction >= early_stopping_window:
            # if we haven't had a reduction in validation loss for `early_stopping_window` epochs, then stop training
            print(f'No reduction in validation loss for {early_stopping_window} epochs. Stopping training...')
            break

    running_test_loss = 0
    correct_test_preds = 0
    # only after finishing training we are testing our model
    with tqdm(test_loader, desc="Testing") as test_pbar:
        for i, (x, y) in enumerate(test_pbar):
            x, y = x.to(device), y.to(device)
            logits = network(x)
            loss = loss_fn(logits, y)

            running_test_loss += loss * len(x)
            correct_test_preds += (logits.argmax(-1) == y).sum()

            if i % 10 == 0:
                num_test_samples_so_far = (i + 1) * len(x)
                val_epoch_pbar.set_postfix(train_loss=running_test_loss.item() / num_test_samples_so_far,
                                           accuracy=correct_test_preds.item() / num_test_samples_so_far * 100)

    avg_test_loss = running_test_loss.item() / num_test_samples
    test_acc = correct_test_preds.item() / num_test_samples * 100

    print(f'Test Set: \tAvg Test Loss: {avg_test_loss:.2f} \tTest Acc: {test_acc:.2f}')




if __name__ == "__main__":
    path = os.getcwd() + r"\..\data\processed\parquet"
    test_loader, train_loader, val_loader = load_data.create_data_loaders(path)
    #longest_sequence = load_data._find_largest_sequence(test_loader, train_loader, val_loader)
    #print("longest sequence in the dataset: " + str(longest_sequence))

    device = "cpu"
    model = ConvNet().to(device)
    train(model, train_loader, val_loader, test_loader, device, epochs=200, lr=1e-3)


