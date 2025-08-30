import torch
from config.settings import Config
import cv2
import numpy as np
import os
from glob import glob

class ChangeDetectionDataset(Dataset):
    """Dataset for training change detection model"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.frame_pairs = self._load_frame_pairs()
        
    def _load_frame_pairs(self):
        """Load frame pairs and their change masks"""
        pairs = []
        
        # Look for frame pairs and corresponding masks
        frame_dirs = glob(os.path.join(self.data_dir, "*"))
        
        for frame_dir in frame_dirs:
            frame1_path = os.path.join(frame_dir, "frame1.jpg")
            frame2_path = os.path.join(frame_dir, "frame2.jpg")
            mask_path = os.path.join(frame_dir, "change_mask.jpg")
            
            if all(os.path.exists(p) for p in [frame1_path, frame2_path, mask_path]):
                pairs.append({
                    'frame1': frame1_path,
                    'frame2': frame2_path,
                    'mask': mask_path
                })
        
        return pairs
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        pair = self.frame_pairs[idx]
        
        # Load images
        frame1 = cv2.imread(pair['frame1'])
        frame2 = cv2.imread(pair['frame2'])
        mask = cv2.imread(pair['mask'], cv2.IMREAD_GRAYSCALE)
        
        # Convert to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            mask = torch.from_numpy(mask / 255.0).float().unsqueeze(0)
        
        # Concatenate frames
        input_tensor = torch.cat([frame1, frame2], dim=0)
        
        return input_tensor, mask

class NumberDataset(Dataset):
    """Dataset for training number classifier"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load number images and labels"""
        samples = []
        
        for digit in range(10):
            digit_dir = os.path.join(self.data_dir, str(digit))
            if os.path.exists(digit_dir):
                image_paths = glob(os.path.join(digit_dir, "*.jpg"))
                for image_path in image_paths:
                    samples.append({
                        'image_path': image_path,
                        'label': digit
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']