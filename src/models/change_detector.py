import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChangeDetectorCNN(nn.Module):
    """Custom CNN model for detecting pixel changes between frames"""
    
    def __init__(self, input_channels=6):  # 3 channels * 2 frames
        super(ChangeDetectorCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Final layer for change map
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        
        # Dropout
        self.dropout = nn.Dropout2d(0.2)
        
    def forward(self, x):
        # x shape: (batch, 6, height, width) - concatenated frames
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Upsample back to original size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Final change map
        change_map = torch.sigmoid(self.final_conv(x))
        
        return change_map
    
    def detect_changes(self, frame1, frame2, threshold=0.3):
        """Detect changes between two frames"""
        self.eval()
        with torch.no_grad():
            # Concatenate frames along channel dimension
            input_tensor = torch.cat([frame1, frame2], dim=1)
            change_map = self.forward(input_tensor)
            
            # Apply threshold
            changes = (change_map > threshold).float()
            
            # Count changed pixels
            change_count = torch.sum(changes).item()
            
            return change_map, changes, change_count

class ChangeDetectorTrainer:
    """Training utilities for the change detector model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    def train_step(self, frame_pairs, targets):
        """Single training step"""
        self.model.train()
        
        frame_pairs = frame_pairs.to(self.device)
        targets = targets.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(frame_pairs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path):
        """Save model state"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load model state"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))