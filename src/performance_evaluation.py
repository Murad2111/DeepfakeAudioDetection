import torch
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import load_data, data_transform_parquet
import nn
import plot_metrics

#-------------------------------Metrics-----------------------------------
def compute_eer(y_true, y_score, positive_lbl=1):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=positive_lbl)
    fnr = 1 - tpr

    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.argmin(abs_diffs)
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    plot_metrics.plot_eer(fpr, fnr, eer, idx_eer, path=os.getcwd()+r"\plots")
    return eer

def compute_tDCF(y_true, y_score, Pfa_asv=0.01, Pmiss_asv=0.01,
                 Cmiss_cm=1, Cfa_cm=1, Ptar=0.5):
    pi_spoof_list = [0.001, 0.05, 0.10, 0.15, 0.20]
    thresholds = np.linspace(min(y_score)-1e-6, max(y_score)+1e-6, 500)
    
    tDCF_dict, min_results = {}, {}
    pi_non = 1 - Ptar

    for pi_spoof in pi_spoof_list:
        tDCF_vals = []
        
        for tau in thresholds:
            preds = (y_score >= tau).astype(int)
            
            Pfa_cm = np.mean(preds[y_true == 1] == 0) if np.sum(y_true == 1) > 0 else 0
            Pmiss_cm = np.mean(preds[y_true == 0] == 1) if np.sum(y_true == 0) > 0 else 0
            
            numerator = (Cmiss_cm * Ptar * Pmiss_cm + Cfa_cm * pi_non * Pfa_cm + Cfa_cm * pi_spoof * Pfa_cm)    
            denominator = min(Cmiss_cm * Ptar, Cfa_cm * (pi_non + pi_spoof))
            
            tDCF = numerator / denominator
            tDCF_vals.append(tDCF)
        
        tDCF_vals = np.array(tDCF_vals)
        tDCF_dict[pi_spoof] = tDCF_vals
        
        min_idx = np.nanargmin(tDCF_vals)
        min_results[pi_spoof] = {
            'min_tDCF': tDCF_vals[min_idx],
            'threshold': thresholds[min_idx]
        }
    
    plot_metrics.plot_tDCF(tDCF_dict, thresholds, path=os.getcwd()+r"\plots")
    
    return min_results

#-------------------------------Evaluate Model-----------------------------------
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            out_prob = model(x).squeeze()
            all_probs.extend(out_prob.cpu().numpy().tolist())
            all_labels.extend(y.squeeze().cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs > 0.5).astype(int)

    cm = confusion_matrix(all_labels, preds.reshape(-1).round())
    auc = roc_auc_score(all_labels, all_probs)
    eer = compute_eer(all_labels, all_probs)
    tDCF_results = compute_tDCF(all_labels, all_probs)

    print(f"ROC AUC: {auc:.4f}")
    print(f"EER: {eer*100:.2f}%")
    print("t-DCF per pi_spoof:")
    for pi, res in tDCF_results.items():
        print(f"pi_spoof={pi}: min_tDCF={res['min_tDCF']:.4f} at threshold={res['threshold']:.4f}")

    plot_metrics.plot_roc_curve(all_labels, all_probs, auc, path=os.getcwd()+r"\plots")
    plot_metrics.plot_pr_curve(all_labels, all_probs, path=os.getcwd()+r"\plots")
    plot_metrics.plot_confusion_matrix(cm, path=os.getcwd()+r"\plots")

if __name__ == "__main__":
    path = os.getcwd() + r"\data\processed\CQCC"
    test_loader, train_loader, val_loader = (
        load_data.create_data_loaders(path, data_transform_parquet.audio_to_cqcc, batch_size=8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # load model
    checkpoint_path = os.getcwd() + r"\checkpoints\model.pt"
    model = nn.BiCorrelationConvNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # evaluate
    evaluate_model(model, test_loader, device)
