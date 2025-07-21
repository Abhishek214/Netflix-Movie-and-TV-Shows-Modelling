"""
Quick fix for your specific checkpoint issue
"""

import torch
import sys
import os
sys.path.append('..')

from backbone import EfficientDetBackbone


def fix_and_export():
    checkpoint_path = 'logs/abhishek/efficientdet-d2_49_8700.pth'
    
    # Load checkpoint to inspect
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Find classifier shape to determine classes
    for name, param in state_dict.items():
        if 'classifier.header.pointwise_conv.conv.weight' in name:
            # [36, 112, 1, 1] means 36/(3*3) = 4 classes (since 3 ratios * 3 scales = 9 anchors per location)
            num_anchors_classes = param.shape[0]  # 36
            num_anchors = 9  # 3 ratios * 3 scales
            num_classes = num_anchors_classes // num_anchors  # 36/9 = 4 classes
            print(f"Detected {num_classes} classes from checkpoint")
            break
    
    # Create model with correct number of classes
    model = EfficientDetBackbone(
        num_classes=num_classes,  # Use detected classes, not 90
        compound_coef=2,
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # Load checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✓ Model loaded with {num_classes} classes")
    
    # Test model
    dummy_input = torch.randn(1, 3, 768, 768)
    with torch.no_grad():
        regression, classification, anchors = model(dummy_input)
        print(f"✓ Model test passed")
        print(f"  Regression: {regression[0].shape}")
        print(f"  Classification: {classification[0].shape}")
    
    # Now export to ONNX (simplified version without post-processing first)
    torch.onnx.export(
        model,
        dummy_input,
        'efficientdet_d2_raw.onnx',
        input_names=['input'],
        output_names=['regression', 'classification', 'anchors'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=11
    )
    
    print("✓ ONNX export completed: efficientdet_d2_raw.onnx")
    return model, num_classes


if __name__ == "__main__":
    fix_and_export()
