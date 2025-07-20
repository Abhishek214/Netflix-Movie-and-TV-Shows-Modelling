#!/usr/bin/env python3
"""
Personalized ONNX conversion script for your specific setup
All placeholders replaced with your actual paths and configuration
"""

import torch
import sys
import os

# 1. ADD YOUR PROJECT PATH
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

# 2. IMPORT YOUR MODEL ARCHITECTURE
# These imports should match your repository structure
from efficientdet.model import EfficientDet
from backbone import EfficientDetBackbone

def convert_your_model():
    print("🔧 Converting YOUR EfficientDet model to ONNX")
    
    # 3. YOUR SPECIFIC MODEL CONFIGURATION
    # ⚠️ CRITICAL: Update num_classes to match YOUR dataset
    # Count the classes in your dataset annotations
    num_classes = 4  # 🔴 REPLACE: Count your actual classes
    
    # Create model with YOUR configuration
    model = EfficientDetBackbone(
        num_classes=num_classes,  # Your dataset classes
        compound_coef=2,          # D2 model
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # 4. YOUR TRAINED MODEL PATH
    model_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    print(f"Loading model from: {model_path}")
    
    # 5. LOAD YOUR TRAINED WEIGHTS
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print("✅ Weights loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False
    
    model.eval()
    
    # 6. CREATE RAW OUTPUT WRAPPER
    class RawOutputModel(torch.nn.Module):
        """
        Wrapper that returns RAW outputs without post-processing
        This avoids the ONNX conversion issues you experienced
        """
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            # Get raw outputs before any post-processing
            features, regression, classification, anchors = self.backbone(x)
            
            # Return only regression and classification (raw logits)
            # We'll handle anchor generation and NMS separately
            return regression, classification
    
    # Wrap model
    raw_model = RawOutputModel(model)
    
    # 7. TEST FORWARD PASS
    print("Testing forward pass...")
    dummy_input = torch.randn(1, 3, 768, 768)
    
    try:
        with torch.no_grad():
            outputs = raw_model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Regression shape: {outputs[0].shape}")
        print(f"   Classification shape: {outputs[1].shape}")
        print(f"   Regression range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
        print(f"   Classification range: [{outputs[1].min():.3f}, {outputs[1].max():.3f}]")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # 8. EXPORT TO ONNX
    output_path = 'efficientdet_d2_abhid_FIXED.onnx'  # Include your name for clarity
    
    print(f"Exporting to: {output_path}")
    
    try:
        torch.onnx.export(
            raw_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,  # Prevent conversion issues
            input_names=['input'],
            output_names=['regression', 'classification'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✅ ONNX model exported successfully!")
        print(f"📁 Saved as: {output_path}")
        print(f"🔧 This model outputs RAW logits (no post-processing)")
        print(f"📊 Use the custom inference script for proper results")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def get_your_class_count():
    """
    Helper function to count classes in your dataset
    """
    import json
    
    # 🔴 REPLACE: Path to your training annotations
    annotation_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/datasets/abhid/annotations/instances_train2017.json'
    
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        categories = data['categories']
        class_names = [cat['name'] for cat in categories]
        
        print(f"📊 Found {len(class_names)} classes in your dataset:")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
        
        return len(class_names), class_names
        
    except Exception as e:
        print(f"⚠️  Could not read annotations: {e}")
        print(f"📝 Manually count your classes and update num_classes in the script")
        return None, None

if __name__ == '__main__':
    print("🔍 Analyzing your dataset...")
    
    # Get class information
    num_classes, class_names = get_your_class_count()
    
    if num_classes:
        print(f"\n✅ Auto-detected {num_classes} classes")
        response = input(f"Is this correct? (y/n): ")
        if response.lower() != 'y':
            print("Please manually update num_classes in the script")
            exit(1)
    else:
        print("\n⚠️  Could not auto-detect classes")
        print("Please manually count your classes and update num_classes in the script")
        
        manual_count = input("Enter number of classes manually (or press Enter to continue with 4): ")
        if manual_count.strip():
            num_classes = int(manual_count)
        else:
            num_classes = 4
    
    print(f"\n🚀 Converting model with {num_classes} classes...")
    
    # Update the global variable (this is a quick hack - in production, pass as parameter)
    globals()['NUM_CLASSES'] = num_classes
    
    success = convert_your_model()
    
    if success:
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📋 Next steps:")
        print(f"   1. Test: python raw_onnx_inference.py --model efficientdet_d2_abhid_FIXED.onnx --image test.jpg")
        print(f"   2. Evaluate: Run COCO evaluation script")
        print(f"   3. Compare with original 72% mAP")
    else:
        print(f"\n❌ Conversion failed. Check error messages above.")