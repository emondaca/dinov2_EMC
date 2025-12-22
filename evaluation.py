import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, eval_test = False, criterion = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    with torch.no_grad():
        #for images, labels in test_loader:
        #    images = images.to(device)
        #    labels = labels.to(device)
        for batch in test_loader:

            images = batch['image_feat'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                #num_batches += 1

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Convert to NumPy arrays
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if eval_test:
        print("\n=== Final Metrics ===")
        print(f"Accuracy:  {acc:.4f}")
    else:
        print(f"Val Accuracy:  {acc:.4f} - F1 Score:  {f1:.4f}")

        val_loss = running_loss / len(test_loader)
        return val_loss, f1

    if eval_test:
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            auc = roc_auc_score(all_labels, all_probs, average="macro")
        except ValueError:
            auc = float("nan")

        print("F1-Score \t Precision \t Recall \t Accuracy \t AUC")
        print(f"{f1:.4f}\t{precision:.4f}\t{recall:.4f}\t{acc:.4f}\t{auc:.4f}")

        return
