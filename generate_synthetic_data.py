#!/usr/bin/env python3
"""
Generate synthetic training data for the models
"""

import cv2
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont
import argparse

def create_change_detection_data(output_dir, num_samples=1000):
    """Generate synthetic data for change detection training"""
    print(f"Generating {num_samples} change detection samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create base image
        base_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Create second image with changes
        changed_img = base_img.copy()
        
        # Add random changes
        num_changes = random.randint(1, 5)
        change_mask = np.zeros((224, 224), dtype=np.uint8)
        
        for _ in range(num_changes):
            # Random rectangle change
            x = random.randint(0, 200)
            y = random.randint(0, 200)
            w = random.randint(10, 50)
            h = random.randint(10, 50)
            
            # Change color in region
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(changed_img, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(change_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Save sample
        sample_dir = os.path.join(output_dir, f"sample_{i:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(sample_dir, "frame1.jpg"), base_img)
        cv2.imwrite(os.path.join(sample_dir, "frame2.jpg"), changed_img)
        cv2.imwrite(os.path.join(sample_dir, "change_mask.jpg"), change_mask)
        
        if i % 100 == 0:
            print(f"Generated {i} samples...")

def create_number_classification_data(output_dir, num_samples_per_digit=500):
    """Generate synthetic number images for classification training"""
    print(f"Generating {num_samples_per_digit} samples per digit...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for digit in range(10):
        digit_dir = os.path.join(output_dir, str(digit))
        os.makedirs(digit_dir, exist_ok=True)
        
        for i in range(num_samples_per_digit):
            # Create image
            img = Image.new('RGB', (32, 32), color=(random.randint(200, 255), 
                                                   random.randint(200, 255), 
                                                   random.randint(200, 255)))
            draw = ImageDraw.Draw(img)
            
            # Draw number with random position and color
            text_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            x_offset = random.randint(-3, 3)
            y_offset = random.randint(-3, 3)
            
            draw.text((8 + x_offset, 4 + y_offset), str(digit), 
                     fill=text_color, font=font)
            
            # Add some noise
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img_path = os.path.join(digit_dir, f"{digit}_{i:06d}.jpg")
            Image.fromarray(img_array).save(img_path)
        
        print(f"Generated samples for digit {digit}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for generated data')
    parser.add_argument('--change_samples', type=int, default=1000,
                       help='Number of change detection samples')
    parser.add_argument('--number_samples', type=int, default=500,
                       help='Number of samples per digit')
    
    args = parser.parse_args()
    
    # Create directories
    change_dir = os.path.join(args.output_dir, 'change_detection')
    number_dir = os.path.join(args.output_dir, 'numbers')
    
    # Generate data
    create_change_detection_data(change_dir, args.change_samples)
    create_number_classification_data(number_dir, args.number_samples)
    
    print("Synthetic data generation complete!")

if __name__ == "__main__":
    main()