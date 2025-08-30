import os

class Config:
    # Model parameters
    CHANGE_DETECTION_THRESHOLD = 0.3
    NUMBER_CONFIDENCE_THRESHOLD = 0.8
    
    # Image processing
    IMAGE_SIZE = (224, 224)
    FRAME_BUFFER_SIZE = 10
    
    # Model paths
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../data/models')
    CHANGE_DETECTOR_PATH = os.path.join(MODEL_DIR, 'change_detector.pth')
    NUMBER_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'number_classifier.pth')
    
    # Video/Camera settings
    CAMERA_INDEX = 0
    FPS = 30
    
    # Display settings
    DISPLAY_DURATION = 2000  # ms
    FONT_SIZE = 32
    FONT_COLOR = (0, 255, 0)  # Green