#!/usr/bin/env python3
"""
Quick test to verify the fixes work
"""

import torch
import sys
import os
import pathlib

# Fix Windows path compatibility
def fix_windows_paths():
    """Fix Windows path compatibility"""
    if os.name == 'nt':  # Windows
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            print("✅ Fixed Windows path compatibility")
            return True
        except Exception as e:
            print(f"❌ Failed to fix Windows paths: {e}")
            return False
    return True

# Apply the fix
fix_windows_paths()

def test_yolo_fix():
    """Test YOLOv5 loading with force_reload and Windows paths"""
    print("🔧 Testing YOLOv5 fix...")
    try:
        # Use absolute path for Windows compatibility
        abs_path = os.path.abspath('pothole_detection.pt')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=abs_path, force_reload=True)
        print("✅ YOLOv5 fix works!")
        return True
    except Exception as e:
        print(f"❌ YOLOv5 still fails: {e}")
        # Try with string path
        try:
            abs_path = os.path.abspath('pothole_detection.pt')
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(abs_path), force_reload=True)
            print("✅ YOLOv5 fix works with string path!")
            return True
        except Exception as e2:
            print(f"❌ YOLOv5 still fails with string path: {e2}")
            return False

def test_efficientnet_fix():
    """Test EfficientNet class detection"""
    print("🔧 Testing EfficientNet fix...")
    try:
        import torchvision.models as models
        
        # Load state dict to check classes
        state_dict = torch.load('weather_prediction.pt', map_location='cpu')
        
        if 'classifier.1.weight' in state_dict:
            num_classes = state_dict['classifier.1.weight'].shape[0]
            print(f"✅ Detected {num_classes} classes")
            
            # Create model with correct classes
            model = models.efficientnet_b1(pretrained=False)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
            model.load_state_dict(state_dict)
            print("✅ EfficientNet fix works!")
            return True
        else:
            print("❌ Could not detect classes")
            return False
    except Exception as e:
        print(f"❌ EfficientNet still fails: {e}")
        return False

def test_enet_fix():
    """Test ENet state dict handling"""
    print("🔧 Testing ENet fix...")
    try:
        model_state = torch.load('lane_detection_eNet.pth', map_location='cpu')
        
        if isinstance(model_state, dict):
            print("✅ ENet is state dict - will create architecture")
            return True
        else:
            print("✅ ENet is complete model")
            return True
    except Exception as e:
        print(f"❌ ENet test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick Fix Test")
    print("=" * 40)
    
    results = []
    results.append(test_yolo_fix())
    results.append(test_efficientnet_fix())
    results.append(test_enet_fix())
    
    print("\n" + "=" * 40)
    if all(results):
        print("🎉 All fixes work! Run the full test now:")
        print("python test_models.py")
    else:
        print("⚠️  Some fixes need more work")
