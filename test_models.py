#!/usr/bin/env python3
"""
Test script to verify all models are loaded correctly
Run this before using the main application
"""

import torch
import cv2
import numpy as np
from PIL import Image
import logging
import os
from pathlib import Path

# Fix path compatibility for Windows
def fix_path_compatibility():
    """Fix path compatibility issues on Windows"""
    try:
        if os.name == 'nt':  # Windows
            import pathlib
            pathlib.PosixPath = pathlib.WindowsPath
            print("✅ Fixed path compatibility for Windows")
    except Exception as e:
        print(f"⚠️  Could not fix path compatibility: {e}")

# Apply the fix immediately
fix_path_compatibility()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if all models can be loaded"""
    print("🔍 Testing Model Loading...")
    print("=" * 50)
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test YOLOv5 models
    print("\n1. Testing YOLOv5 Models:")
    
    # Test pothole model
    print("Testing pothole_detection.pt...")
    try:
        # Convert to absolute path
        abs_path = os.path.abspath('pothole_detection.pt')
        pothole_model = torch.hub.load('ultralytics/yolov5', 'custom', path=abs_path, force_reload=True)
        pothole_model.to(device)
        pothole_model.eval()
        print("✅ Pothole detection model loaded successfully")
    except Exception as e:
        print(f"❌ Pothole detection model failed: {e}")
        # Try with string path
        try:
            pothole_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(abs_path), force_reload=True)
            pothole_model.to(device)
            pothole_model.eval()
            print("✅ Pothole detection model loaded successfully (string path)")
        except Exception as e2:
            print(f"❌ Pothole detection model failed with string path: {e2}")
    
    # Test sign model
    print("Testing sign_detection.pt...")
    try:
        # Convert to absolute path
        abs_path = os.path.abspath('sign_detection.pt')
        sign_model = torch.hub.load('ultralytics/yolov5', 'custom', path=abs_path, force_reload=True)
        sign_model.to(device)
        sign_model.eval()
        print("✅ Sign detection model loaded successfully")
    except Exception as e:
        print(f"❌ Sign detection model failed: {e}")
        # Try with string path
        try:
            sign_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(abs_path), force_reload=True)
            sign_model.to(device)
            sign_model.eval()
            print("✅ Sign detection model loaded successfully (string path)")
        except Exception as e2:
            print(f"❌ Sign detection model failed with string path: {e2}")
    
    # Test ENet model
    print("\n2. Testing ENet Model:")
    try:
        # Load using the proper ENet-SAD architecture
        from enet_sad import ENet_SAD
        
        # Create model with proper architecture
        lane_model = ENet_SAD(backbone='enet', sad=True, num_classes=2)
        
        # Load state dict
        state_dict = torch.load('lane_detection_eNet.pth', map_location=device)
        lane_model.load_state_dict(state_dict)
        lane_model.to(device)
        lane_model.eval()
        
        print("✅ Lane detection model loaded successfully (ENet-SAD architecture)")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 256, 512).to(device)
        with torch.no_grad():
            output = lane_model(dummy_input)
        print(f"✅ Lane model inference successful - output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Lane detection model failed: {e}")
    
    # Test EfficientNet model
    print("\n3. Testing EfficientNet Model:")
    try:
        import torchvision.models as models
        
        # First, load the state dict to check the actual number of classes
        state_dict = torch.load('weather_prediction.pt', map_location=device)
        
        # Determine the correct number of classes from the state dict
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
            print(f"Detected {num_classes} weather classes from model")
        else:
            num_classes = 4  # Default fallback
            print(f"Could not determine number of classes, using default: {num_classes}")
        
        # Create model with correct number of classes
        weather_model = models.efficientnet_b1(pretrained=False)
        weather_model.classifier[1] = torch.nn.Linear(weather_model.classifier[1].in_features, num_classes)
        
        # Load the state dict
        weather_model.load_state_dict(state_dict)
        weather_model.to(device)
        weather_model.eval()
        print("✅ Weather prediction model loaded successfully")
    except Exception as e:
        print(f"❌ Weather prediction model failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Model loading test completed!")
    print("If all models show ✅, you're ready to run the main application!")

def test_unified_system():
    """Test the unified detection system"""
    print("\n🔧 Testing Unified Detection System...")
    print("=" * 50)
    
    try:
        from unified_detection_app import UnifiedDetectionSystem
        
        # Create the system
        detector = UnifiedDetectionSystem()
        
        # Check if ready
        if detector.is_ready():
            print("✅ Unified Detection System is ready!")
            print(f"Device: {detector.device}")
            
            # Test with a dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            print("\n🧪 Testing with dummy image...")
            results = detector.process_frame_parallel(dummy_image)
            
            print("Results:")
            for key, value in results.items():
                if key == 'weather' and isinstance(value, tuple):
                    print(f"  {key}: {value[0]} (confidence: {value[1]:.2f})")
                elif key == 'lanes':
                    print(f"  {key}: {'Detected' if value is not None else 'Failed'}")
                else:
                    print(f"  {key}: {len(value) if isinstance(value, list) else value}")
            
            print("✅ Unified system test passed!")
        else:
            print("❌ Unified Detection System is not ready!")
            
    except Exception as e:
        print(f"❌ Unified system test failed: {e}")

if __name__ == "__main__":
    print("🚗 Unified Detection System - Model Test")
    print("=" * 50)
    
    # Test individual models
    test_model_loading()
    
    # Test unified system
    test_unified_system()
    
    print("\n🎉 All tests completed!")
    print("Run 'streamlit run unified_detection_app.py' to start the application!")
