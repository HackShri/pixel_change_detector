#!/usr/bin/env python3
"""
Training script for both change detection and number classification models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from src.models import ChangeDetectorCNN, ChangeDetectorTrainer
from src.models import NumberClassifierCNN, NumberClassifierTrainer
from src.utils import ChangeDetectionDataset, NumberDataset
from src.config import Config

def train_change_detector(data_dir, epochs=50, batch_size=16):
    """Train the change detection model"""
    print("Training Change Detection Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and dataloader
    dataset = ChangeDetectionDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model and trainer
    model = ChangeDetectorCNN()
    trainer = ChangeDetectorTrainer(model, device)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (frame_pairs, targets) in enumerate(dataloader):
            loss = trainer.train_step(frame_pairs, targets)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    trainer.save_model(config.CHANGE_DETECTOR_PATH)
    print(f"Change detector model saved to {config.CHANGE_DETECTOR_PATH}")

def train_number_classifier(data_dir, epochs=30, batch_size=32):
    """Train the number classification model"""
    print("Training Number Classification Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and dataloader
    dataset = NumberDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model and trainer
    model = NumberClassifierCNN()
    trainer = NumberClassifierTrainer(model, device)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            loss = trainer.train_step(images, labels)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    trainer.save_model(config.NUMBER_CLASSIFIER_PATH)
    print(f"Number classifier model saved to {config.NUMBER_CLASSIFIER_PATH}")

def main():
    parser = argparse.ArgumentParser(description='Train pixel change detection models')
    parser.add_argument('--model', choices=['change_detector', 'number_classifier', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--change_data_dir', type=str, default='./data/change_detection',
                       help='Directory containing change detection training data')
    parser.add_argument('--number_data_dir', type=str, default='./data/numbers',
                       help='Directory containing number classification training data')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.model in ['change_detector', 'both']:
        train_change_detector(args.change_data_dir, args.epochs, args.batch_size)
    
    if args.model in ['number_classifier', 'both']:
        train_number_classifier(args.number_data_dir, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()