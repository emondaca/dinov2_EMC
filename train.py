import torch
from evaluation import evaluate_model
import torch.nn as nn
from sklearn.metrics import f1_score
from model import RadDINOLastBlockClassifier, radDino_LastBlock_PlusHead

import matplotlib.pyplot as plt
import numpy as np
import copy
def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()


def train_model(train_loader, val_loader, num_classes,pos_weight, save_path,RadDino_src,RadDinoWeights,head_dir,classWeighted = True,  num_epochs=20, early_Stopping = 60, radDinoType = 1, raddinoHead = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if raddinoHead:
        model = radDino_LastBlock_PlusHead(768,RadDino_src, RadDinoWeights, head_dir, num_classes=num_classes).to(device)
    else:
        model = RadDINOLastBlockClassifier(RadDino_src, RadDinoWeights,num_classes, in_dim = 768, radDinoType = radDinoType).to(device)

    if classWeighted:
        pos_weight = pos_weight.to(device)
        print("class weights Tensor: ", pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Class Weights On")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Class Weights Off")

    # Optimizer
    if raddinoHead:
        trainable_params = [
            {"params": model.last_block.parameters(), "lr": 1e-5},
            {"params": model.final_norm.parameters(), "lr": 1e-5},
            {"params": (p for p in model.rad_dino_head_gh.mlp.parameters() if p.requires_grad), "lr": 1e-4},
            {"params": (p for p in model.rad_dino_head_gh.last_layer.parameters() if p.requires_grad), "lr": 1e-3}
            #{"params": model.classifier.parameters(), "lr": 1e-3}
        ]
    else:
        trainable_params = [
            {"params": model.last_block.parameters(), "lr": 1e-5}, #1e-3
            {"params": model.final_norm.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-3}
        ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
    
    """trainable_params = [
        {"params": model.encoder.parameters(), "lr": 1e-4}, 
        {"params": [model.cls_token, model.pos_embedding], "lr": 5e-4},
        {"params": model.head.parameters(), "lr": 1e-4}, 
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-4)
    print(f'Lr_head = {1e-4} - cls_token,pos_embedding  = {5e-5} - encoder = {1e-4}')"""

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       
        factor=0.6,      
        patience=15,       
        min_lr=5e-6       
    )

    train_losses = []
    val_losses = []
    best_f1 = 0.0
    best_model = None

    patience_counter = 0

    print("Start the loop Training ...")
    test = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_preds = []
        epoch_labels = []

        for batch in train_loader:

            images = batch['image_feat'].to(device)
            labels = batch['label'].to(device)
            if test == 0:
                print("Image in training loop shape:", images.shape)
                test += 1
            #B, P, D = images.shape   # Np should be 9
            #image_flat = images.reshape(B * P, D)

            optimizer.zero_grad()
            outputs = model(images)  

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            epoch_labels.append(labels.cpu())
            epoch_preds.append(preds.cpu())

        train_loss = running_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds).numpy()
        epoch_labels = torch.cat(epoch_labels).numpy()

        training_f1 = f1_score(epoch_labels, epoch_preds, average="macro", zero_division=0)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_loss:.4f} - Training F1: {training_f1:.4f}")


        # ---- Validation ----
        val_loss, f1 = evaluate_model(model, val_loader, criterion = criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else: 
            patience_counter += 1

        if patience_counter >= early_Stopping:
            print(f"Stopping early -------------- ")
            break  # Stop training early

    save_loss_plot(train_losses, val_losses, save_path)
    return best_model