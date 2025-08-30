import unittest
import torch
import numpy as np
from src.models import ChangeDetectorCNN, NumberClassifierCNN, PixelChangeDetectionPipeline

class TestModels(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.change_detector = ChangeDetectorCNN()
        self.number_classifier = NumberClassifierCNN()
        
    def test_change_detector_forward(self):
        """Test change detector forward pass"""
        # Create dummy input (batch_size=1, channels=6, height=224, width=224)
        dummy_input = torch.randn(1, 6, 224, 224)
        
        output = self.change_detector(dummy_input)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 224, 224))
        
        # Check output range (should be 0-1 due to sigmoid)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_number_classifier_forward(self):
        """Test number classifier forward pass"""
        # Create dummy input (batch_size=1, channels=3, height=32, width=32)
        dummy_input = torch.randn(1, 3, 32, 32)
        
        output = self.number_classifier(dummy_input)
        
        # Check output shape (10 classes)
        self.assertEqual(output.shape, (1, 10))
    
    def test_change_detection(self):
        """Test change detection functionality"""
        frame1 = torch.randn(1, 3, 224, 224)
        frame2 = torch.randn(1, 3, 224, 224)
        
        change_map, changes, change_count = self.change_detector.detect_changes(
            frame1, frame2, threshold=0.5
        )
        
        # Check types and shapes
        self.assertIsInstance(change_count, (int, float))
        self.assertEqual(change_map.shape, (1, 1, 224, 224))
        self.assertEqual(changes.shape, (1, 1, 224, 224))
    
    def test_number_prediction(self):
        """Test number prediction functionality"""
        # Create dummy image patch
        image_patch = torch.randn(1, 3, 32, 32)
        
        predicted_number, confidence = self.number_classifier.predict_number(
            image_patch, confidence_threshold=0.1  # Low threshold for testing
        )
        
        # Check output types
        if predicted_number is not None:
            self.assertIsInstance(predicted_number, int)
            self.assertTrue(0 <= predicted_number <= 9)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)

class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = PixelChangeDetectionPipeline(device='cpu')
        
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline.change_detector)
        self.assertIsNotNone(self.pipeline.number_classifier)
        self.assertIsNotNone(self.pipeline.image_processor)
    
    def test_process_frame(self):
        """Test frame processing"""
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results = self.pipeline.process_frame(frame)
        
        # Check result structure
        required_keys = ['frame', 'changes_detected', 'change_regions', 
                        'detected_numbers', 'change_count']
        for key in required_keys:
            self.assertIn(key, results)
        
        self.assertIsInstance(results['changes_detected'], bool)
        self.assertIsInstance(results['change_regions'], list)
        self.assertIsInstance(results['detected_numbers'], list)
        self.assertIsInstance(results['change_count'], (int, float))

if __name__ == '__main__':
    unittest.main()