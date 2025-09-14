import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_curve(fpr, tpr, auc, path):
    fpr, tpr, _ = roc_curve(fpr, tpr)
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(path+r"\roc_curve.png")

def plot_pr_curve(labels, probs, path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.clf()
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(path+r"\pr_curve.png")

def plot_confusion_matrix(cm, path):
    plt.clf()
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix", fontsize=14)
    plt.colorbar()
    class_names = ['Real', 'Fake']
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(path+r"\confusion_matrix.png")

def plot_eer(fpr, fnr, eer, idx_err, path):
    plt.clf()
    plt.figure()
    plt.plot(fpr, fnr, label="FNR vs FPR")
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter(fpr[idx_err], fnr[idx_err], color="red", label=f"EER = {eer*100:.2f}%")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("Equal Error Rate[EER] Plot")
    plt.legend()
    plt.savefig(path+r"\eer_plot.png")

def plot_tDCF(tDCF_vals, thresholds, path):
    plt.clf()
    plt.figure(figsize=(10, 6))

    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'd', 'x']
    
    for i, (pi_spoof, tDCF_vals) in enumerate(tDCF_vals.items()):
        min_idx = tDCF_vals.argmin()
        plt.plot(thresholds, tDCF_vals, linestyle=linestyles[i % len(linestyles)],
                 label=f'pi_spoof={pi_spoof}')
        plt.scatter(thresholds[min_idx], tDCF_vals[min_idx], marker=markers[i % len(markers)],
                    s=50, zorder=5)

    plt.xlabel('Threshold')
    plt.ylabel('t-DCF')
    plt.title('Tandem Detection Cost Function (t-DCF) Plot')
    plt.grid(True)
    plt.legend()
    plt.savefig(path+r"\tDCF_plot.png")