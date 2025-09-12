import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, early_stop, path, num_epochs):
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.axvline(x=early_stop, color="red", linestyle="--", alpha=0.7)
    plt.text(early_stop + 0.1, max(max(train_losses), max(val_losses)) * 0.95,
             "Early Stop", color="red")

    batches_per_epoch = int(len(train_losses) / num_epochs)
    epoch_starts = [i * batches_per_epoch for i in range(num_epochs)]
    plt.scatter(epoch_starts, [train_losses[i] for i in epoch_starts],
                color="blue", marker="D", s=50, label="Epoch start")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.savefig(path+r"\loss.png")


def plot_acc(train_acc, val_acc, early_stop, path, num_epochs):
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(x=early_stop, color="red", linestyle="--", alpha=0.7)
    plt.text(early_stop + 0.1, max(max(train_acc), max(val_acc)) * 0.7,
             "Early Stop", color="red")

    batches_per_epoch = int(len(train_acc) / num_epochs)
    epoch_starts = [i * batches_per_epoch for i in range(num_epochs)]
    plt.scatter(epoch_starts, [train_acc[i] for i in epoch_starts],
                color="blue", marker="D", s=50, label="Epoch start")

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.ylim(60, 100)
    plt.savefig(path+r"\acc.png")