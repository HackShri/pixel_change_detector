import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NumberClassifierCNN(nn.Module):
    """Custom CNN for classifying numbers in changed regions"""
    
    def __init__(self, num_classes=10):  # 0-9 digits
        super(NumberClassifierCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_number(self, image_patch, confidence_threshold=0.8):
        """Predict number in image patch"""
        self.eval()
        with torch.no_grad():
            output = self.forward(image_patch)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            if confidence.item() > confidence_threshold:
                return predicted.item(), confidence.item()
            else:
                return None, confidence.item()

class NumberClassifierTrainer:
    """Training utilities for the number classifier"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def save_model(self, path):
        """Save model state"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model state"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
    def train_step(self, images, labels):
        """Single training step"""
        self.model.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data_loader):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return 100 * correct / total