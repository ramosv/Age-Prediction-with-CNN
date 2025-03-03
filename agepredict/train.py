import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from agepredict.utils.dataset import AgesDataset
from agepredict.utils.transforms import get_train_transforms, get_val_transforms
from agepredict.models.resnet import RestNet
from torch.optim.lr_scheduler import StepLR
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(
    csv_path,
    img_dir,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    checkpoint_path='output/checkpoints/resnet_best.pth',
    early_stop_patience=3,
    step_size=5,
    gamma=0.1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    full_dataset = AgesDataset(csv_file=csv_path, img_dir=img_dir)
    n_total = len(full_dataset)
    logger.info(f"Full dataset size: {n_total}")

    val_split = 0.2
    val_size = int(n_total * val_split)
    train_size = n_total - val_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    train_subset.dataset.transform = get_train_transforms()
    val_subset.dataset.transform = get_val_transforms()
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    # model, loss, optimizer and scheduler
    # if want to use cnn change this like 
    model = RestNet(num_classes=1, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    best_val_rmse = float('inf')
    epochs_without_improvement = 0
    
    best_model_path = os.path.join(os.path.dirname(checkpoint_path), "temp_best_model.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    # main trainning loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, ages in train_loader:
            images = images.to(device)
            ages = ages.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, ages in val_loader:
                images = images.to(device)
                ages = ages.to(device)
                outputs = model(images).view(-1)
                
                loss = criterion(outputs, ages)
                val_loss += loss.item() * images.size(0)
                
                val_preds.extend(outputs.cpu().numpy().tolist())
                val_targets.extend(ages.cpu().numpy().tolist())
        
        val_loss /= len(val_loader.dataset)
        mse = mean_squared_error(val_targets, val_preds)
        rmse = np.sqrt(mse)
        
        # updates on training progress at each epoch
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val RMSE: {rmse:.4f} - LR: {scheduler.get_last_lr()[0]:.5f}")
        
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stop_patience:
            logger.info("Early stopping triggered!")
            break
        
        scheduler.step()
    
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Best model (Val RMSE={best_val_rmse:.4f}) saved to {checkpoint_path}")
    
    return best_val_rmse

def train_model_with_validation(
    train_csv,
    val_csv,
    img_dir,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    checkpoint_path='output/checkpoints/fold.pth',
    early_stop_patience=3,
    step_size=5,
    gamma=0.1
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = AgesDataset(csv_file=train_csv, img_dir=img_dir)
    val_dataset = AgesDataset(csv_file=val_csv,   img_dir=img_dir)

    train_dataset.transform = get_train_transforms()
    val_dataset.transform   = get_val_transforms()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    # model, loss, optimizer and scheduler
    # if want to use cnn change this like 
    model = RestNet(num_classes=1, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    best_val_rmse = float('inf')
    epochs_without_improvement = 0
    
    best_model_path = os.path.join(os.path.dirname(checkpoint_path), "temp_best_model.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, ages in train_loader:
            images = images.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()
            outputs = model(images).view(-1)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, ages in val_loader:
                images = images.to(device)
                ages = ages.to(device)
                outputs = model(images).view(-1)

                loss = criterion(outputs, ages)
                val_loss += loss.item() * images.size(0)

                val_preds.extend(outputs.cpu().numpy().tolist())
                val_targets.extend(ages.cpu().numpy().tolist())

        val_loss /= len(val_loader.dataset)
        mse = mean_squared_error(val_targets, val_preds)
        rmse = np.sqrt(mse)

        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val RMSE: {rmse:.4f} - LR: {scheduler.get_last_lr()[0]:.5f}")

        # early stop check
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            logger.info("Early stopping triggered!")
            break

        scheduler.step()

    # load best and return
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Best model (Val RMSE={best_val_rmse:.4f}) saved to {checkpoint_path}")

    return best_val_rmse

def cross_val_training(
    csv_path,
    img_dir,
    n_splits=5,
    random_state=42,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    checkpoint_root="output/checkpoints",
):

    df = pd.read_csv(csv_path)
    n_total = len(df)
    logger.info(f"Total samples in cleaned dataset: {n_total}")

    # split the dataset
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # loop over the folds and train
    fold_rmse_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_total))):
        logger.info(f"\nFold {fold_idx+1}/{n_splits}")

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df   = df.iloc[val_idx].copy().reset_index(drop=True)

        train_csv = f"temp_train_fold{fold_idx}.csv"
        val_csv   = f"temp_val_fold{fold_idx}.csv"

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        fold_ckpt_path = f"{checkpoint_root}/fold_{fold_idx}.pth"

        # now call train_model_with_validation
        fold_rmse = train_model_with_validation(
            train_csv=train_csv,
            val_csv=val_csv,
            img_dir=img_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            checkpoint_path=fold_ckpt_path,
            early_stop_patience=3,
            step_size=5,
            gamma=0.1
        )
        
        # append the fold rmse to the list
        fold_rmse_list.append(fold_rmse)

    # this average is the final metric and a much better representation of the model's performance
    avg_rmse = np.mean(fold_rmse_list)
    logger.info(f"\nCross-Validation average RMSE: {avg_rmse:.4f}")
    return avg_rmse
