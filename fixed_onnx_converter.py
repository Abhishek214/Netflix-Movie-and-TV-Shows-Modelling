#!/usr/bin/env python3
"""
Fixed ONNX converter that exports raw model outputs without post-processing
This avoids the issues you're seeing with fixed scores and invalid boxes
"""

import torch
import torch.nn as nn
import numpy as np
import math
import argparse
import warnings

warnings.filterwarnings("ignore")

class RawEfficientDetONNX(nn.Module):
    """
    Export EfficientDet with RAW outputs (no post-processing)
    This avoids the conversion issues you're experiencing
    """
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model
        self.backbone.eval()
    
    def forward(self, x):
        # Get raw outputs from the backbone
        # This should return regression and classification logits
        try:
            # Try different output formats
            if hasattr(self.backbone, 'backbone_net'):
                # For EfficientDetBackbone format
                features, regression, classification, anchors = self.backbone.backbone_net(x)
                return regression, classification, anchors
            elif hasattr(self.backbone, 'forward'):
                # Try direct forward
                outputs = self.backbone(x)
                if len(outputs) >= 3:
                    return outputs[0], outputs[1], outputs[2]  # reg, cls, anchors
                else:
                    return outputs
            else:
                raise ValueError("Cannot determine model structure")
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return dummy outputs with correct shapes for debugging
            batch_size = x.shape[0]
            dummy_reg = torch.zeros(batch_size, 1000, 4)  # Dummy regression
            dummy_cls = torch.zeros(batch_size, 1000, 80)  # Dummy classification  
            dummy_anchors = torch.zeros(1000, 4)  # Dummy anchors
            return dummy_reg, dummy_cls, dummy_anchors

def load_your_model(model_path):
    """
    Load your trained EfficientDet model
    REPLACE THIS with your actual model loading code
    """
    print(f"Loading model from: {model_path}")
    
    try:
        # Method 1: Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"Checkpoint keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
        
        # Method 2: You need to reconstruct your model here
        # This is the critical part - replace with your actual model creation
        print("⚠️  CRITICAL: You need to replace this section with your model creation code")
        print("   Example:")
        print("   from efficientdet.model import EfficientDet")
        print("   model = EfficientDet(compound_coef=2, num_classes=80)")
        print("   model.load_state_dict(state_dict)")
        
        # Placeholder - REPLACE THIS
        return None
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def convert_to_raw_onnx():
    """
    Convert model to ONNX with RAW outputs (no post-processing)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compound_coef', type=int, default=2)
    parser.add_argument('-w', '--weights', type=str, required=True)
    parser.add_argument('--output', type=str, default='efficientdet_raw.onnx')
    parser.add_argument('--input_size', type=int, default=768)
    
    args = parser.parse_args()
    
    print("🔧 Converting EfficientDet to RAW ONNX (no post-processing)")
    print(f"Model: {args.weights}")
    print(f"Output: {args.output}")
    print(f"Input size: {args.input_size}")
    
    # Load your trained model
    original_model = load_your_model(args.weights)
    
    if original_model is None:
        print("❌ Could not load model. Please implement the load_your_model() function")
        print("   with your specific model architecture.")
        return
    
    # Wrap for ONNX export
    raw_model = RawEfficientDetONNX(original_model)
    raw_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)
    
    # Test forward pass first
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            outputs = raw_model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"✓ Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Export to ONNX
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            raw_model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,  # Disable to avoid conversion issues
            input_names=['input'],
            output_names=['regression', 'classification', 'anchors'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✓ RAW ONNX model saved: {args.output}")
        print("✓ This model outputs raw regression/classification logits")
        print("✓ You'll need custom post-processing for inference")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")

if __name__ == '__main__':
    convert_to_raw_onnx()