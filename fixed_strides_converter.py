#!/usr/bin/env python3
"""
Fixed ONNX converter that matches your exact training architecture
Solves the "strides has incorrect size" error
"""

import torch
import sys
import os
import json

# Add your project path
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

def load_training_config(project_name='abhid'):
    """
    Load the EXACT configuration used during training
    """
    config_path = f'/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/projects/{project_name}.yml'
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded training config from: {config_path}")
        print(f"   Project: {config.get('project_name', 'Unknown')}")
        print(f"   Classes: {len(config.get('obj_list', []))}")
        print(f"   Anchors scales: {config.get('anchors_scales', 'Not specified')}")
        print(f"   Anchors ratios: {config.get('anchors_ratios', 'Not specified')}")
        
        return config
        
    except Exception as e:
        print(f"⚠️  Could not load config from {config_path}: {e}")
        print("Using default configuration...")
        return None

def get_exact_model_architecture(config):
    """
    Create the EXACT model architecture used during training
    """
    try:
        # Import the correct model classes
        from efficientdet.model import EfficientDet
        from backbone import EfficientDetBackbone
        
        # Get configuration from training config
        if config:
            num_classes = len(config.get('obj_list', []))
            anchors_scales = eval(config.get('anchors_scales', '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'))
            anchors_ratios = eval(config.get('anchors_ratios', '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'))
        else:
            # Fallback defaults
            num_classes = 4  # Update this manually if needed
            anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        
        print(f"Creating model with:")
        print(f"  - Classes: {num_classes}")
        print(f"  - Compound coef: 2 (D2)")
        print(f"  - Anchor scales: {anchors_scales}")
        print(f"  - Anchor ratios: {anchors_ratios}")
        
        # Create model with EXACT training configuration
        model = EfficientDetBackbone(
            num_classes=num_classes,
            compound_coef=2,
            ratios=anchors_ratios,
            scales=anchors_scales
        )
        
        return model, num_classes
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("Trying alternative model creation...")
        
        # Alternative: Try the direct EfficientDet class
        try:
            from efficientdet.model import EfficientDet
            
            model = EfficientDet(
                num_classes=num_classes if 'num_classes' in locals() else 4,
                compound_coef=2
            )
            
            return model, num_classes if 'num_classes' in locals() else 4
            
        except Exception as e2:
            print(f"❌ Alternative model creation also failed: {e2}")
            return None, None

def load_checkpoint_safely(model, checkpoint_path):
    """
    Safely load checkpoint with detailed error reporting
    """
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("✅ Found 'model' key in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Found 'state_dict' key in checkpoint")
        else:
            state_dict = checkpoint
            print("✅ Using checkpoint directly as state_dict")
        
        # Get model's expected keys
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Check for key mismatches
        missing_in_checkpoint = model_keys - checkpoint_keys
        extra_in_checkpoint = checkpoint_keys - model_keys
        
        if missing_in_checkpoint:
            print(f"⚠️  Keys missing in checkpoint: {len(missing_in_checkpoint)}")
            for key in list(missing_in_checkpoint)[:5]:  # Show first 5
                print(f"   - {key}")
        
        if extra_in_checkpoint:
            print(f"⚠️  Extra keys in checkpoint: {len(extra_in_checkpoint)}")
            for key in list(extra_in_checkpoint)[:5]:  # Show first 5
                print(f"   - {key}")
        
        # Load with strict=False to handle mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("✅ Checkpoint loaded perfectly!")
        else:
            print(f"⚠️  Loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def test_model_forward_pass(model, input_size=768):
    """
    Test model forward pass to catch strides issues early
    """
    try:
        print(f"Testing forward pass with input size {input_size}...")
        
        model.eval()
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✅ Forward pass successful!")
        print(f"   Number of outputs: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"   Output {i}: {output.shape}")
            else:
                print(f"   Output {i}: {type(output)}")
        
        return True, outputs
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        
        # If it's a strides error, provide specific guidance
        if "strides" in str(e).lower():
            print("\n🔧 STRIDES ERROR DETECTED!")
            print("This means the model architecture doesn't match training.")
            print("Possible fixes:")
            print("1. Check if you're using the correct compound_coef (should be 2 for D2)")
            print("2. Verify anchor configuration matches training")
            print("3. Ensure EfficientNet backbone version is correct")
        
        return False, None

class ArchitectureMatchedRawModel(torch.nn.Module):
    """
    Raw model wrapper that preserves exact training architecture
    """
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model
        
    def forward(self, x):
        try:
            # Call the model exactly as during training
            outputs = self.model(x)
            
            # Handle different output formats
            if len(outputs) == 4:
                # Standard format: features, regression, classification, anchors
                features, regression, classification, anchors = outputs
                return regression, classification
            elif len(outputs) == 3:
                # Alternative format: regression, classification, anchors
                regression, classification, anchors = outputs
                return regression, classification
            elif len(outputs) == 2:
                # Already regression, classification
                return outputs[0], outputs[1]
            else:
                # Unknown format
                print(f"⚠️  Unexpected output format: {len(outputs)} outputs")
                return outputs[0], outputs[1] if len(outputs) > 1 else outputs[0]
                
        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            raise

def convert_with_architecture_matching():
    """
    Convert model ensuring exact architecture match
    """
    print("🔧 ARCHITECTURE-MATCHED ONNX CONVERSION")
    print("="*50)
    
    # 1. Load training configuration
    config = load_training_config('abhid')  # Your project name
    
    # 2. Create exact model architecture
    model, num_classes = get_exact_model_architecture(config)
    
    if model is None:
        print("❌ Could not create model architecture")
        return False
    
    # 3. Load trained weights
    checkpoint_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    if not load_checkpoint_safely(model, checkpoint_path):
        return False
    
    # 4. Test forward pass
    success, outputs = test_model_forward_pass(model)
    
    if not success:
        print("❌ Model forward pass failed - cannot convert to ONNX")
        return False
    
    # 5. Create raw output wrapper
    raw_model = ArchitectureMatchedRawModel(model)
    
    # 6. Export to ONNX
    output_path = 'efficientdet_d2_architecture_matched.onnx'
    
    try:
        print(f"Exporting to ONNX: {output_path}")
        
        dummy_input = torch.randn(1, 3, 768, 768)
        
        torch.onnx.export(
            raw_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['regression', 'classification'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print("✅ ONNX export successful!")
        print(f"📁 Model saved as: {output_path}")
        
        # 7. Verify ONNX model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
            
            # Test ONNX inference
            dummy_input_np = dummy_input.numpy()
            onnx_outputs = session.run(None, {'input': dummy_input_np})
            
            print("✅ ONNX model verification successful!")
            print(f"   ONNX output shapes: {[out.shape for out in onnx_outputs]}")
            
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

def debug_architecture_mismatch():
    """
    Debug helper to identify architecture mismatches
    """
    print("\n🔍 DEBUGGING ARCHITECTURE MISMATCH")
    print("="*40)
    
    # Check if PyYAML is available
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not found. Install with: pip install PyYAML")
        return
    
    # Check project config
    project_config = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/projects/abhid.yml'
    
    if os.path.exists(project_config):
        print(f"✅ Project config found: {project_config}")
        
        try:
            with open(project_config, 'r') as f:
                config = yaml.safe_load(f)
            
            print("📋 Training configuration:")
            for key, value in config.items():
                print(f"   {key}: {value}")
                
        except Exception as e:
            print(f"❌ Could not read config: {e}")
    else:
        print(f"❌ Project config not found: {project_config}")
        print("   This might be causing the architecture mismatch")
    
    # Check model file
    model_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                print(f"📋 Model state keys (first 10): {list(model_state.keys())[:10]}")
            
        except Exception as e:
            print(f"❌ Could not load checkpoint: {e}")
    else:
        print(f"❌ Model file not found: {model_path}")

if __name__ == '__main__':
    # First, debug any architecture issues
    debug_architecture_mismatch()
    
    print("\n" + "="*60)
    
    # Then attempt conversion
    success = convert_with_architecture_matching()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Architecture-matched ONNX model created")
        print(f"📋 Next steps:")
        print(f"   1. Test: python raw_onnx_inference.py --model efficientdet_d2_architecture_matched.onnx --image test.jpg")
        print(f"   2. This should resolve the strides error")
        
    else:
        print(f"\n❌ CONVERSION FAILED")
        print(f"📋 Troubleshooting steps:")
        print(f"   1. Verify project config exists: projects/abhid.yml")
        print(f"   2. Check model checkpoint is valid")
        print(f"   3. Ensure training and conversion use same architecture")
        print(f"   4. Install missing dependencies: pip install PyYAML")