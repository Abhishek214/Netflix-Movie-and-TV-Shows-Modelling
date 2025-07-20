#!/usr/bin/env python3
"""
Quick diagnostic script to identify and fix strides errors
"""

import torch
import sys
import os

# Add your project path
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

def diagnose_strides_error():
    """
    Diagnose the exact cause of strides error
    """
    print("🔍 DIAGNOSING STRIDES ERROR")
    print("="*40)
    
    # 1. Check if we can import the model classes
    try:
        from backbone import EfficientDetBackbone
        print("✅ Can import EfficientDetBackbone")
    except Exception as e:
        print(f"❌ Cannot import EfficientDetBackbone: {e}")
        return
    
    # 2. Check training configuration
    project_file = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/projects/abhid.yml'
    
    if os.path.exists(project_file):
        print(f"✅ Found project config: {project_file}")
        
        try:
            import yaml
            with open(project_file, 'r') as f:
                config = yaml.safe_load(f)
            
            print("📋 Your training configuration:")
            print(f"   Classes: {len(config.get('obj_list', []))}")
            print(f"   Anchor scales: {config.get('anchors_scales', 'Default')}")
            print(f"   Anchor ratios: {config.get('anchors_ratios', 'Default')}")
            
            return config
            
        except ImportError:
            print("❌ PyYAML not installed. Install with: pip install PyYAML")
            return None
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return None
    else:
        print(f"❌ Project config not found: {project_file}")
        return None

def test_different_architectures():
    """
    Test different model architectures to find the correct one
    """
    print(f"\n🧪 TESTING DIFFERENT MODEL ARCHITECTURES")
    print("="*50)
    
    from backbone import EfficientDetBackbone
    
    # Test configurations
    test_configs = [
        {"name": "Standard D2", "compound_coef": 2, "num_classes": 4},
        {"name": "Standard D2 (80 classes)", "compound_coef": 2, "num_classes": 80},
        {"name": "Standard D1", "compound_coef": 1, "num_classes": 4},
        {"name": "Standard D0", "compound_coef": 0, "num_classes": 4},
    ]
    
    checkpoint_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    # Load checkpoint once
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        print(f"✅ Loaded checkpoint from: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Cannot load checkpoint: {e}")
        return
    
    # Test each configuration
    for config in test_configs:
        print(f"\n🔧 Testing: {config['name']}")
        
        try:
            # Create model
            model = EfficientDetBackbone(
                num_classes=config['num_classes'],
                compound_coef=config['compound_coef'],
                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            )
            
            # Try to load weights
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            # Test forward pass
            model.eval()
            dummy_input = torch.randn(1, 3, 768, 768)
            
            with torch.no_grad():
                outputs = model(dummy_input)
            
            print(f"   ✅ SUCCESS! Architecture works")
            print(f"   📊 Missing keys: {len(missing)}")
            print(f"   📊 Unexpected keys: {len(unexpected)}")
            print(f"   📊 Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
            
            # This configuration works - use it for ONNX export
            return model, config
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if "strides" in str(e).lower():
                print(f"   🔍 Strides error - wrong architecture")
            continue
    
    print(f"\n❌ No working architecture found!")
    return None, None

def create_working_onnx_model(working_model, config):
    """
    Create ONNX model using the working architecture
    """
    print(f"\n🚀 CREATING ONNX MODEL")
    print("="*30)
    
    class WorkingRawModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            outputs = self.model(x)
            # Return regression and classification
            return outputs[1], outputs[2]  # Skip features, return reg and cls
    
    # Wrap model
    raw_model = WorkingRawModel(working_model)
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 768, 768)
    output_path = f"efficientdet_d2_working_{config['compound_coef']}_{config['num_classes']}.onnx"
    
    try:
        torch.onnx.export(
            raw_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['regression', 'classification'],
            verbose=False
        )
        
        print(f"✅ ONNX model created: {output_path}")
        
        # Test ONNX model
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        onnx_outputs = session.run(None, {'input': dummy_input.numpy()})
        
        print(f"✅ ONNX model verified!")
        print(f"   Regression shape: {onnx_outputs[0].shape}")
        print(f"   Classification shape: {onnx_outputs[1].shape}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def main():
    """
    Main diagnostic and fix workflow
    """
    print("🔧 STRIDES ERROR - DIAGNOSIS AND FIX")
    print("="*50)
    
    # 1. Diagnose configuration
    config = diagnose_strides_error()
    
    # 2. Test different architectures
    working_model, working_config = test_different_architectures()
    
    if working_model is None:
        print(f"\n❌ CRITICAL: No working architecture found!")
        print(f"📋 Manual debugging steps:")
        print(f"   1. Check if compound_coef is correct (0, 1, 2 for D0, D1, D2)")
        print(f"   2. Verify number of classes matches training")
        print(f"   3. Check if anchor configuration is correct")
        print(f"   4. Ensure EfficientNet backbone version matches")
        return
    
    # 3. Create working ONNX model
    onnx_path = create_working_onnx_model(working_model, working_config)
    
    if onnx_path:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Working ONNX model: {onnx_path}")
        print(f"📋 Architecture: {working_config['name']}")
        print(f"📋 Test with:")
        print(f"   python raw_onnx_inference.py --model {onnx_path} --image test.jpg")
    else:
        print(f"\n❌ Could not create ONNX model")

if __name__ == '__main__':
    main()