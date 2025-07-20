#!/usr/bin/env python3
"""
Quick fix for Abhid's model - tries the most likely solutions
"""

import torch
import sys
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

from backbone import EfficientDetBackbone

def quick_fix_abhid_model():
    """
    Quick fix based on debug analysis showing 4 vs 12 classes mismatch
    """
    print("⚡ QUICK FIX FOR ABHID'S 4-CLASS MODEL")
    print("="*40)
    
    checkpoint_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    # The debug output showed "Possible classes: 12" but config says 4
    # Let's try both possibilities
    
    test_configs = [
        {"name": "12 classes (from debug)", "classes": 12, "coef": 2},
        {"name": "4 classes (from config)", "classes": 4, "coef": 2},
        {"name": "4 classes D1", "classes": 4, "coef": 1},
        {"name": "4 classes D0", "classes": 4, "coef": 0},
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        
        try:
            # Create model
            model = EfficientDetBackbone(
                num_classes=config['classes'],
                compound_coef=config['coef']
            )
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
            
            # Test forward pass
            model.eval()
            test_input = torch.randn(1, 3, 768, 768)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            print(f"   ✅ SUCCESS! This configuration works!")
            print(f"   📊 Missing keys: {len(missing)}")
            print(f"   📊 Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
            
            # Export to ONNX immediately
            class QuickRawModel(torch.nn.Module):
                def __init__(self, backbone):
                    super().__init__()
                    self.backbone = backbone
                
                def forward(self, x):
                    outputs = self.backbone(x)
                    return outputs[1], outputs[2]  # regression, classification
            
            raw_model = QuickRawModel(model)
            dummy_input = torch.randn(1, 3, 768, 768)
            
            onnx_filename = f"efficientdet_abhid_FIXED_{config['classes']}classes.onnx"
            
            torch.onnx.export(
                raw_model, dummy_input, onnx_filename,
                export_params=True, opset_version=11, do_constant_folding=False,
                input_names=['input'], output_names=['regression', 'classification']
            )
            
            print(f"   🎉 ONNX MODEL CREATED: {onnx_filename}")
            
            # Test ONNX model
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])
            onnx_outputs = session.run(None, {'input': dummy_input.numpy()})
            
            print(f"   ✅ ONNX verification successful!")
            print(f"   📊 ONNX shapes: {[out.shape for out in onnx_outputs]}")
            
            print(f"\n🎉 SOLUTION FOUND!")
            print(f"✅ Working configuration: {config['name']}")
            print(f"✅ ONNX model: {onnx_filename}")
            print(f"\n📋 Test it with:")
            print(f"python raw_onnx_inference.py --model {onnx_filename} --image test.jpg")
            
            return onnx_filename
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if "strides" in str(e).lower():
                print(f"   🚨 Strides error with {config['name']}")
            continue
    
    print(f"\n❌ No quick fix worked")
    print(f"Run the comprehensive fix: python abhid_specific_fix.py")
    return None

if __name__ == '__main__':
    quick_fix_abhid_model()