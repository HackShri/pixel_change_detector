import torch
import numpy as np
import cv2
from .change_detector import ChangeDetectorCNN
from .number_classifier import NumberClassifierCNN
from utils.image_processor import ImageProcessor
from config.settings import Config

class PixelChangeDetectionPipeline:
    """Main pipeline combining change detection and number classification"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.config = Config()
        
        # Initialize models
        self.change_detector = ChangeDetectorCNN()
        self.number_classifier = NumberClassifierCNN()
        
        # Load pretrained weights if available
        self._load_models()
        
        # Image processor
        self.image_processor = ImageProcessor()
        
        # Frame buffer for tracking changes
        self.frame_buffer = []
        self.previous_frame = None
        
    def _load_models(self):
        """Load pretrained model weights"""
        try:
            if torch.cuda.is_available() and self.device == 'cuda':
                self.change_detector.load_state_dict(
                    torch.load(self.config.CHANGE_DETECTOR_PATH)
                )
                self.number_classifier.load_state_dict(
                    torch.load(self.config.NUMBER_CLASSIFIER_PATH)
                )
            else:
                self.change_detector.load_state_dict(
                    torch.load(self.config.CHANGE_DETECTOR_PATH, map_location='cpu')
                )
                self.number_classifier.load_state_dict(
                    torch.load(self.config.NUMBER_CLASSIFIER_PATH, map_location='cpu')
                )
            print("Models loaded successfully")
        except FileNotFoundError:
            print("Pretrained models not found. Using randomly initialized weights.")
            print("Train the models first or use the training scripts.")
    
    def process_frame(self, current_frame):
        """Process a single frame and detect changes"""
        results = {
            'frame': current_frame,
            'changes_detected': False,
            'change_regions': [],
            'detected_numbers': [],
            'change_count': 0
        }
        
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return results
        
        # Preprocess frames
        prev_tensor = self.image_processor.preprocess_frame(self.previous_frame)
        curr_tensor = self.image_processor.preprocess_frame(current_frame)
        
        # Detect changes
        change_map, changes, change_count = self.change_detector.detect_changes(
            prev_tensor, curr_tensor, self.config.CHANGE_DETECTION_THRESHOLD
        )
        
        results['change_count'] = change_count
        
        if change_count > 0:
            results['changes_detected'] = True
            
            # Find change regions
            change_regions = self._find_change_regions(changes.squeeze().cpu().numpy())
            results['change_regions'] = change_regions
            
            # Classify numbers in change regions
            detected_numbers = self._classify_numbers_in_regions(
                current_frame, change_regions
            )
            results['detected_numbers'] = detected_numbers
        
        # Update previous frame
        self.previous_frame = current_frame.copy()
        
        return results
    
    def _find_change_regions(self, change_mask):
        """Find bounding boxes of change regions"""
        # Convert to uint8 for OpenCV operations
        change_mask_uint8 = (change_mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            change_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small changes
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour)
                })
        
        return regions
    
    def _classify_numbers_in_regions(self, frame, regions):
        """Classify numbers in detected change regions"""
        detected_numbers = []
        
        for region in regions:
            x, y, w, h = region['bbox']
            
            # Extract region of interest
            roi = frame[y:y+h, x:x+w]
            
            if roi.size > 0:
                # Preprocess ROI for number classification
                roi_tensor = self.image_processor.preprocess_roi(roi)
                
                # Classify number
                predicted_number, confidence = self.number_classifier.predict_number(
                    roi_tensor, self.config.NUMBER_CONFIDENCE_THRESHOLD
                )
                
                if predicted_number is not None:
                    detected_numbers.append({
                        'number': predicted_number,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
        
        return detected_numbers
    
    def reset(self):
        """Reset the pipeline state"""
        self.frame_buffer.clear()
        self.previous_frame = None