import torch.nn as nn
from tqdm.auto import tqdm
import torch
import numpy as np
import os
import load_data, data_transform_parquet
import gc
import plotting

#-----------------------------classes--------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, inout_channels, hidden_channels, kernel_size, act_fn=nn.ReLU()):
        super().__init__()
        pad = 0
        if type(kernel_size) == int:
            pad = int(kernel_size/2)
        elif type(kernel_size) == tuple:
            pad = (int(kernel_size[0]/2), int(kernel_size[1]/2))
        self.conv = nn.Sequential(
            nn.Conv2d(inout_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
            act_fn,
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
            act_fn,
            nn.Conv2d(hidden_channels, inout_channels, kernel_size=kernel_size, padding=pad),
            act_fn
        )

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2), :x.size(3)]
        return x + out  # residual connection



class ConvNet(torch.nn.Module):
    """
    change this later
    """
    def __init__(self):
        super().__init__()
        act_fn = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))#not symmetric because our input dimensions are (13,400) worst case
        kernel_size=(1,2)
        self.convLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=kernel_size), act_fn,
            pool,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size), act_fn,
            ResidualBlock(32, 32, kernel_size, act_fn=act_fn),
            pool,
            ResidualBlock(32, 32, kernel_size, act_fn=act_fn),#maybe overkill
            pool,
            ResidualBlock(32, 32, kernel_size, act_fn=act_fn),  # maybe overkill
            pool,
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size), act_fn,
            pool,
        )

        self.fullyConnected = nn.Sequential(
            #nn.Linear(4000, 120),  # out_channels*x_cur*y_cur
            nn.Linear(76800, 120),
            act_fn,
            nn.Linear(120, 84),
            act_fn,
            nn.Linear(84, 1),
            nn.Sigmoid(),#for probability as output
        )

    def forward(self, x):
        time_x = self.convLayers(x)
        #freq_x = self.convFrequencyLayers(x)

        # flatten the convolution results
        time_x = time_x.view(time_x.size(0), -1)  #out_channels*x_cur*y_cur
        #freq_x = freq_x.view(freq_x.size(0), -1)

        x = time_x
        #x= torch.cat((time_x, freq_x), dim=1)

        x = self.fullyConnected(x)

        return x


class BiCorrelationConvNet(torch.nn.Module):
    """
    change this later
    """
    def __init__(self):
        super().__init__()
        act_fn = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))#not symmetric because our input dimensions are (13,400) worst case
        time_kernel_size=(1,2)
        self.timeCorrelationLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=time_kernel_size), act_fn,
            pool,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=time_kernel_size), act_fn,
            #ResidualBlock(32, 32, time_kernel_size, act_fn=act_fn),
            pool,
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=time_kernel_size), act_fn,
            pool,
        )
        self.timeToVector = nn.Linear(65600, 1000)

        coefficient_kernel_size=(6,1)
        self.coefficientCorrelationLayers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=coefficient_kernel_size), act_fn,
            pool,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=coefficient_kernel_size), act_fn,
            #ResidualBlock(32, 32, coefficient_kernel_size, act_fn=act_fn),
            pool,
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=coefficient_kernel_size), act_fn,
            pool,
        )
        self.coeffiecientToVector = nn.Linear(16480, 1000)

        self.correlationNetwork = nn.Sequential(
            #nn.Linear(4000, 120),  # out_channels*x_cur*y_cur
            nn.Linear(2000, 240),
            act_fn,
            nn.Linear(240, 1),
            nn.Sigmoid(),#for probability as output
        )

    def forward(self, x):
        time_x = self.timeCorrelationLayers(x)
        # flatten the convolution results
        time_x = time_x.view(time_x.size(0), -1)  #out_channels*x_cur*y_cur
        vec_time = self.timeToVector(time_x)

        freq_x = self.coefficientCorrelationLayers(x)
        freq_x = freq_x.view(time_x.size(0), -1)
        vec_freq = self.coeffiecientToVector(freq_x)

        x= torch.cat((vec_time, vec_freq), dim=1)

        x = self.correlationNetwork(x)

        return x


#------------------------------utility functions----------------------------------------
def train(network, train_loader, val_loader, device, checkpoint_path, epochs=200, lr=1e-4, regularization=1e-4):
    """
    copied from exercise for now
    """
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    early_stopping_iter = 0
    final_epoch = 0

    loss_fn = torch.nn.BCELoss() #our network should output a single probability imo
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=regularization)

    best_model_valid_loss = np.Inf
    num_epochs_without_val_loss_reduction = 0
    early_stopping_window = 3

    running_train_loss = 0
    running_val_loss = 0
    correct_train_preds = 0
    correct_val_preds = 0
    num_train_samples_so_far = 0
    num_val_samples_so_far = 0

    #-----------------------------------------train------------------------
    model.train()
    for epoch in range(epochs):
        final_epoch = epoch

        # iterate the training data
        with tqdm(train_loader, desc="Training") as train_epoch_pbar:
            for i, (x, y) in enumerate(train_epoch_pbar):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                out_prob = network(x)
                loss = loss_fn(out_prob, y)
                loss.backward()
                optimizer.step()

                running_train_loss += loss * len(x)
                preds = (out_prob > 0.5).float().squeeze()
                correct_train_preds += (preds == y.squeeze()).sum()
                #print((preds == y))

                num_train_samples_so_far += len(x)

                if i % 10 == 0:
                    train_loss = running_train_loss.item() / num_train_samples_so_far
                    accuracy = (correct_train_preds.item() / num_train_samples_so_far) * 100
                    train_loss_list.append(train_loss)
                    train_acc_list.append(accuracy)
                    train_epoch_pbar.set_postfix(train_loss=train_loss,
                                                 accuracy=accuracy)


        # --------------------------------------val----------------------------
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc="Validating") as val_epoch_pbar:
                for i, (x, y) in enumerate(val_epoch_pbar):
                    x, y = x.to(device), y.to(device)
                    out_prob = network(x)
                    loss = loss_fn(out_prob, y)

                    running_val_loss += loss * len(x)
                    preds = (out_prob > 0.5).float().squeeze()
                    correct_val_preds += (preds == y.squeeze()).sum()
                    num_val_samples_so_far += len(x)

                    if i % 10 == 0:
                        val_loss = running_val_loss.item() / num_val_samples_so_far,
                        accuracy = correct_val_preds.item() / num_val_samples_so_far * 100
                        val_loss_list.append(val_loss)
                        val_acc_list.append(accuracy)
                        val_epoch_pbar.set_postfix(val_loss=val_loss,
                                                   accuracy=accuracy)

        avg_train_loss = running_train_loss.item() / num_train_samples_so_far
        train_acc = correct_train_preds.item() / num_train_samples_so_far * 100
        avg_val_loss = running_val_loss.item() / num_val_samples_so_far
        val_acc = correct_val_preds.item() / num_val_samples_so_far * 100

        print(
            f'Epoch {epoch}: \tAvg Train Loss: {avg_train_loss:.2f} \tTrain Acc: {train_acc:.2f} \tAvg Val Loss: {avg_val_loss:.2f} \tVal Acc: {val_acc:.2f}')

        # perform early stopping if necessary
        if avg_val_loss < best_model_valid_loss:
            print(f'Validation loss decreased ({best_model_valid_loss:.6f} --> {avg_val_loss:.6f}).  Saving model ...')
            torch.save(network.state_dict(), checkpoint_path+ r'\model.pt')
            num_epochs_without_val_loss_reduction = 0
            best_model_valid_loss = avg_val_loss

            early_stopping_iter = len(train_loss_list)-1
        else:
            num_epochs_without_val_loss_reduction += 1

        if num_epochs_without_val_loss_reduction >= early_stopping_window:
            # if we haven't had a reduction in validation loss for `early_stopping_window` epochs, then stop training
            print(f'No reduction in validation loss for {early_stopping_window} epochs. Stopping training...')
            break

    return (np.array(train_loss_list), np.array(train_acc_list), np.array(val_loss_list),
            np.array(val_acc_list), early_stopping_iter, final_epoch+1)


def test_nn(network, test_loader, device):
    loss_fn = torch.nn.BCELoss()  # our network should output a single probability imo
    running_test_loss = 0
    correct_test_preds = 0
    num_test_samples_so_far = 0

    model.eval()
    with torch.no_grad(): #dont track gradient info
        with tqdm(test_loader, desc="Testing") as test_pbar:
            for i, (x, y) in enumerate(test_pbar):
                x, y = x.to(device), y.to(device)
                out_prob = network(x)
                loss = loss_fn(out_prob, y)

                running_test_loss += loss * len(x)
                preds = (out_prob > 0.5).float().squeeze()
                correct_test_preds += (preds == y.squeeze()).sum()
                num_test_samples_so_far += len(x)

                if i % 10 == 0:
                    test_pbar.set_postfix(test_loss=running_test_loss.item() / num_test_samples_so_far,
                                               accuracy=correct_test_preds.item() / num_test_samples_so_far * 100)

    avg_test_loss = running_test_loss.item() / num_test_samples_so_far
    test_acc = correct_test_preds.item() / num_test_samples_so_far * 100

    print(f'Test Set: \tAvg Test Loss: {avg_test_loss:.2f} \tTest Acc: {test_acc:.2f}')




if __name__ == "__main__":
    path = os.getcwd() + r"\..\data\processed\CQCC"
    test_loader, train_loader, val_loader = (
        load_data.create_data_loaders(path, data_transform_parquet.audio_to_cqcc, batch_size=8))
    #longest_sequence = load_data._find_largest_sequence(test_loader, train_loader, val_loader)
    #print("longest sequence in the dataset: " + str(longest_sequence))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: " + str(device))
    #model = ConvNet().to(device)
    model = BiCorrelationConvNet().to(device)

    #---------------------------------train---------------------------
    #comment this+plot out if you just want to test
    checkpoint_path = os.getcwd() + r"\..\checkpoints"
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, early_stopping_iter, epoch_nr = (
        train(model, train_loader, val_loader, device, checkpoint_path, epochs=200, lr=1e-4))  #training

    #---------------------------------plot-----------------------------
    plot_path = os.getcwd() + r"\..\plots"
    plotting.plot_loss(train_loss_list, val_loss_list, early_stopping_iter, plot_path, epoch_nr)
    plotting.plot_acc(train_acc_list, val_acc_list, early_stopping_iter, plot_path, epoch_nr)

    #------------------------------cleanup-----------------------------
    del train_loader, val_loader  #free up memory
    torch.cuda.empty_cache()
    gc.collect()

    #-----------------------------test-------------------------------
    #model = ConvNet().to(device)  # init the model
    model.load_state_dict(torch.load(checkpoint_path + r"\model.pt"))
    test_nn(model, test_loader, device)