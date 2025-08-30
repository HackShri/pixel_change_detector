# 🎯 Real-Time Pixel Change Detection with Number Recognition

A computer vision system that detects pixel-level changes in video streams and automatically recognizes numbers in changed regions using custom neural networks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-green.svg)](https://opencv.org/)

## 🌟 Features

- 🔍 **Real-time pixel change detection** using custom CNN models
- 🔢 **Automatic number recognition** in changed regions
- 📹 **Live camera feed processing** with visualization
- 🎥 **Video file batch processing** 
- 🧠 **Model-based approach** - no traditional CV libraries for core detection
- ⚡ **GPU acceleration** support
- 📊 **Performance monitoring** with FPS counter
- 🎨 **Rich visualization** with heatmaps and overlays

## 🎬 Demo

![Demo GIF](demo.gif) <!-- Add a demo GIF if you have one -->

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for real-time detection)
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pixel-change-detector.git
   cd pixel-change-detector

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Generate training data**:
   ```bash
   python generate_synthetic_data.py
4. **Train the models**:
   ```bash
   python train_models.py

5. **Run the application**:
   ```bash
   python run.py

 ### Programmatic Usage
   ```bash
   from src.models.pipeline import PixelChangeDetectionPipeline

   # Initialize
   pipeline = PixelChangeDetectionPipeline()

   # Process frame
   results = pipeline.process_frame(your_frame)
   print(f"Detected numbers: {results['detected_numbers']}")
