# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, Blur
from albumentations.pytorch import ToTensorV2
from model import UNet
from dataset import RobotDataset
from util import EarlyStopping, plot_training_curves, plot_confusion_matrix, save_training_data

class RobotSegmentation:
    def __init__(self, dataset_dir, n_classes=3, batch_size=8, epochs=250, lr=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_dir = dataset_dir
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = UNet(n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20)
        self.train_loader, self.val_loader = self.load_data()

        self.train_losses, self.val_losses = [], []
        self.iou_scores, self.precisions, self.recalls, self.f1_scores = [], [], [], []

    def load_data(self):
        transforms = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            Blur(blur_limit=3, p=0.5),
            ToTensorV2(),
        ])
        annotation_path = os.path.join(self.dataset_dir, 'annotations.json')
        image_dir = os.path.join(self.dataset_dir, 'images')
        full_dataset = RobotDataset(image_dir, annotation_path, img_size=512, augmentations=transforms)
        train_size = int(0.75 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True), \
               DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.epochs):
            train_loss = self.train_model()
            val_loss, avg_iou, avg_precision, avg_recall, avg_f1, all_preds, all_targets = self.validate_model()
            self.scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.iou_scores.append(avg_iou)
            self.precisions.append(avg_precision)
            self.recalls.append(avg_recall)
            self.f1_scores.append(avg_f1)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            if val_loss < early_stopping.val_loss_min:
                torch.save(self.model.state_dict(), 'results/checkpoint.pth')
                early_stopping.val_loss_min = val_loss
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                break
        plot_confusion_matrix(all_targets, all_preds, self.n_classes)
        plot_training_curves(self.train_losses, self.val_losses)
        self.save_model("results/Multiclass_2_model_5_6_24.pth")
        save_training_data(self.train_losses, self.val_losses, self.iou_scores, self.precisions, self.recalls, self.f1_scores)

    def train_model(self):
        self.model.train()
        losses = []
        for imgs, masks in self.train_loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device).long()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def validate_model(self):
        self.model.eval()
        losses, ious, avg_precision, avg_recall, avg_f1 = [], [], [], [], []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device).long()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)
                losses.append(loss.item())
                # Call iou and metric calculation functions here
        return np.mean(losses), np.mean(ious, axis=0), avg_precision, avg_recall, avg_f1, all_preds, all_targets

    def save_model(self, filepath):
        torch.save(self.model, filepath)
        print(f"Model saved to {filepath}")
