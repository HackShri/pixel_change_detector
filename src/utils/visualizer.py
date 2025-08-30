import cv2
import numpy as np
from config.settings import Config

class Visualizer:
    """Utilities for visualizing detection results"""
    
    def __init__(self):
        self.config = Config()
        
    def draw_change_regions(self, frame, change_regions):
        """Draw bounding boxes around change regions"""
        viz_frame = frame.copy()
        
        for region in change_regions:
            x, y, w, h = region['bbox']
            cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add area text
            area_text = f"Area: {region['area']:.0f}"
            cv2.putText(viz_frame, area_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return viz_frame
    
    def draw_detected_numbers(self, frame, detected_numbers):
        """Draw detected numbers on frame"""
        viz_frame = frame.copy()
        
        for detection in detected_numbers:
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            number = detection['number']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Draw number and confidence
            text = f"{number} ({confidence:.2f})"
            cv2.putText(viz_frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Highlight center
            cv2.circle(viz_frame, (center_x, center_y), 5, (255, 0, 0), -1)
        
        return viz_frame
    
    def create_change_heatmap(self, change_map):
        """Create heatmap visualization of changes"""
        # Convert to numpy if tensor
        if torch.is_tensor(change_map):
            change_map = change_map.squeeze().cpu().numpy()
        
        # Normalize to 0-255
        heatmap = (change_map * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        
        return heatmap_colored
    
    def overlay_results(self, frame, results):
        """Create comprehensive visualization of all results"""
        viz_frame = frame.copy()
        
        if results['changes_detected']:
            # Draw change regions
            viz_frame = self.draw_change_regions(viz_frame, results['change_regions'])
            
            # Draw detected numbers
            viz_frame = self.draw_detected_numbers(viz_frame, results['detected_numbers'])
            
            # Add summary text
            summary_text = f"Changes: {results['change_count']:.0f}, Numbers: {len(results['detected_numbers'])}"
            cv2.putText(viz_frame, summary_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return viz_frame