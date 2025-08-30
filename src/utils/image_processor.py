import cv2
import torch
import numpy as np
from torchvision import transforms
from config.settings import Config

class ImageProcessor:
    """Utilities for image preprocessing and postprocessing"""
    
    def __init__(self):
        self.config = Config()
        
        # Transform for change detection
        self.change_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for number classification
        self.number_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_frame(self, frame):
        """Preprocess frame for change detection"""
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply transforms
        tensor = self.change_transform(frame_rgb)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_roi(self, roi):
        """Preprocess region of interest for number classification"""
        # Convert BGR to RGB if needed
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        else:
            roi_rgb = roi
        
        # Apply transforms
        tensor = self.number_transform(roi_rgb)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def enhance_contrast(self, image):
        """Enhance image contrast for better detection"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def denoise_image(self, image):
        """Remove noise from image"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def extract_number_regions(self, image, change_mask):
        """Extract potential number regions from image using change mask"""
        # Find contours in change mask
        contours, _ = cv2.findContours(
            change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area for numbers
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Numbers typically have certain aspect ratios
                if 0.3 <= aspect_ratio <= 3.0:
                    roi = image[y:y+h, x:x+w]
                    regions.append({
                        'roi': roi,
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return regions