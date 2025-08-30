import unittest
import numpy as np
import cv2
from src.utils import ImageProcessor, Visualizer

class TestImageProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = ImageProcessor()
        
    def test_preprocess_frame(self):
        """Test frame preprocessing"""
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = self.processor.preprocess_frame(frame)
        
        # Check output is tensor with correct shape
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape[0], 1)  # Batch dimension
        self.assertEqual(processed.shape[1], 3)  # Channels
    
    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        enhanced = self.processor.enhance_contrast(frame)
        
        self.assertEqual(enhanced.shape, frame.shape)
        self.assertEqual(enhanced.dtype, frame.dtype)

class TestVisualizer(unittest.TestCase):
    
    def setUp(self):
        self.visualizer = Visualizer()
        
    def test_draw_change_regions(self):
        """Test drawing change regions"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = [
            {'bbox': (10, 10, 50, 50), 'area': 2500},
            {'bbox': (100, 100, 30, 30), 'area': 900}
        ]
        
        viz_frame = self.visualizer.draw_change_regions(frame, regions)
        
        self.assertEqual(viz_frame.shape, frame.shape)
        # Frame should be modified (not all zeros anymore)
        self.assertTrue(np.any(viz_frame > 0))
    
    def test_create_change_heatmap(self):
        """Test heatmap creation"""
        # Create dummy change map
        change_map = np.random.rand(224, 224)
        
        heatmap = self.visualizer.create_change_heatmap(change_map)
        
        self.assertEqual(len(heatmap.shape), 3)  # Should be colored
        self.assertEqual(heatmap.shape[2], 3)    # RGB channels

if __name__ == '__main__':
    unittest.main()