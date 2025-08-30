import cv2
import torch
import numpy as np
from models.pipeline import PixelChangeDetectionPipeline
from utils.visualizer import Visualizer
from config.settings import Config
import time

class PixelChangeDetector:
    """Main application class"""
    
    def __init__(self):
        self.config = Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize pipeline and visualizer
        self.pipeline = PixelChangeDetectionPipeline(device=self.device)
        self.visualizer = Visualizer()
        
        # Video capture
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)
        
        # Performance tracking
        self.fps_counter = 0
        self.start_time = time.time()
        
    def run_realtime_detection(self):
        """Run real-time pixel change detection"""
        print("Starting real-time pixel change detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Process frame
            results = self.pipeline.process_frame(frame)
            
            # Visualize results
            viz_frame = self.visualizer.overlay_results(frame, results)
            
            # Add FPS counter
            self._add_fps_counter(viz_frame)
            
            # Display results
            cv2.imshow('Pixel Change Detection', viz_frame)
            
            # Print detected numbers
            if results['detected_numbers']:
                numbers_text = ", ".join([
                    f"{det['number']} ({det['confidence']:.2f})" 
                    for det in results['detected_numbers']
                ])
                print(f"Detected numbers: {numbers_text}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'saved_frame_{int(time.time())}.jpg', viz_frame)
                print("Frame saved!")
            elif key == ord('r'):
                self.pipeline.reset()
                print("Pipeline reset!")
        
        self._cleanup()
    
    def process_video_file(self, video_path, output_path=None):
        """Process a video file for change detection"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.pipeline.process_frame(frame)
            
            # Store detections
            if results['detected_numbers']:
                total_detections.extend(results['detected_numbers'])
            
            # Visualize and save if output specified
            if output_path:
                viz_frame = self.visualizer.overlay_results(frame, results)
                out.write(viz_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        
        print(f"Processing complete. Total frames: {frame_count}")
        print(f"Total number detections: {len(total_detections)}")
        
        return total_detections
    
    def _add_fps_counter(self, frame):
        """Add FPS counter to frame"""
        self.fps_counter += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.start_time = time.time()
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up.")

def main():
    """Main function to run the application"""
    detector = PixelChangeDetector()
    
    # Choose mode
    print("Pixel Change Detection System")
    print("1. Real-time detection (camera)")
    print("2. Process video file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        detector.run_realtime_detection()
    elif choice == "2":
        video_path = input("Enter video file path: ")
        output_path = input("Enter output path (optional, press Enter to skip): ")
        if not output_path:
            output_path = None
        detector.process_video_file(video_path, output_path)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()