#!/usr/bin/env python3
"""
Specific fix for Abhid's 4-class EfficientDet model
Addresses the 4 vs 12 classes mismatch causing strides error
"""

import torch
import sys
import yaml

# Add your project path
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

def load_exact_training_config():
    """
    Load the exact configuration from your abhid.yml file
    """
    print("🔍 LOADING EXACT TRAINING CONFIGURATION")
    print("="*45)
    
    config_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/projects/abhid.yml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config from: {config_path}")
        print(f"📋 Project: {config.get('project_name', 'Unknown')}")
        print(f"📋 Classes: {len(config.get('obj_list', []))}")
        print(f"📋 Class list: {config.get('obj_list', [])}")
        print(f"📋 Anchor scales: {config.get('anchors_scales', 'Default')}")
        print(f"📋 Anchor ratios: {config.get('anchors_ratios', 'Default')}")
        
        return config
        
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        # Fallback to manual configuration
        print("Using manual fallback configuration...")
        return {
            'obj_list': ['class1', 'class2', 'class3', 'class4'],  # 4 classes
            'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
            'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
        }

def create_model_with_exact_config(config):
    """
    Create model using the exact training configuration
    """
    print(f"\n🔧 CREATING MODEL WITH EXACT CONFIG")
    print("="*40)
    
    try:
        from backbone import EfficientDetBackbone
        
        # Extract configuration
        num_classes = len(config.get('obj_list', []))
        anchors_scales = eval(config.get('anchors_scales', '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'))
        anchors_ratios = eval(config.get('anchors_ratios', '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'))
        
        print(f"🎯 Model configuration:")
        print(f"   Classes: {num_classes}")
        print(f"   Compound coef: 2 (D2)")
        print(f"   Anchor scales: {anchors_scales}")
        print(f"   Anchor ratios: {anchors_ratios}")
        
        # Create model with exact configuration
        model = EfficientDetBackbone(
            num_classes=num_classes,
            compound_coef=2,
            ratios=anchors_ratios,
            scales=anchors_scales
        )
        
        print(f"✅ Model created successfully")
        return model, num_classes
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None, None

def load_weights_with_debugging(model, checkpoint_path):
    """
    Load weights with detailed debugging to understand mismatches
    """
    print(f"\n🔍 LOADING WEIGHTS WITH DEBUGGING")
    print("="*40)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        
        print(f"✅ Checkpoint loaded: {len(state_dict)} keys")
        
        # Get model's expected state dict
        model_state_dict = model.state_dict()
        
        # Analyze key mismatches
        model_keys = set(model_state_dict.keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing_in_checkpoint = model_keys - checkpoint_keys
        extra_in_checkpoint = checkpoint_keys - model_keys
        
        print(f"📊 Key analysis:")
        print(f"   Model expects: {len(model_keys)} keys")
        print(f"   Checkpoint has: {len(checkpoint_keys)} keys")
        print(f"   Missing in checkpoint: {len(missing_in_checkpoint)}")
        print(f"   Extra in checkpoint: {len(extra_in_checkpoint)}")
        
        # Check specific problematic layers
        problematic_keys = []
        for key in model_keys:
            if key in state_dict:
                model_shape = model_state_dict[key].shape
                checkpoint_shape = state_dict[key].shape
                
                if model_shape != checkpoint_shape:
                    problematic_keys.append((key, model_shape, checkpoint_shape))
        
        if problematic_keys:
            print(f"\n⚠️  Shape mismatches found:")
            for key, model_shape, checkpoint_shape in problematic_keys[:5]:  # Show first 5
                print(f"   {key}:")
                print(f"      Model expects: {model_shape}")
                print(f"      Checkpoint has: {checkpoint_shape}")
        
        # Load weights (non-strict to handle mismatches)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"\n📊 Loading results:")
        print(f"   Missing keys: {len(missing_keys)}")
        print(f"   Unexpected keys: {len(unexpected_keys)}")
        
        if len(missing_keys) < 10:  # Show details if not too many
            print(f"   Missing keys: {missing_keys}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False

def test_with_different_approaches(config):
    """
    Try different approaches to create a working model
    """
    print(f"\n🧪 TESTING DIFFERENT APPROACHES")
    print("="*35)
    
    checkpoint_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    approaches = [
        {
            "name": "Exact config (4 classes)",
            "num_classes": 4,
            "compound_coef": 2
        },
        {
            "name": "Inferred config (12 classes)", 
            "num_classes": 12,  # Based on your debug output
            "compound_coef": 2
        },
        {
            "name": "D1 with 4 classes",
            "num_classes": 4,
            "compound_coef": 1
        },
        {
            "name": "D0 with 4 classes",
            "num_classes": 4,
            "compound_coef": 0
        }
    ]
    
    for approach in approaches:
        print(f"\n🔧 Trying: {approach['name']}")
        
        try:
            from backbone import EfficientDetBackbone
            
            # Create model
            model = EfficientDetBackbone(
                num_classes=approach['num_classes'],
                compound_coef=approach['compound_coef'],
                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            )
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"   📊 Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # Test forward pass
            model.eval()
            test_input = torch.randn(1, 3, 768, 768)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            print(f"   ✅ SUCCESS! Forward pass works")
            print(f"   📊 Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
            
            return model, approach
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if "strides" in str(e).lower():
                print(f"   🚨 Strides error with {approach['name']}")
            continue
    
    print(f"\n❌ No approach worked")
    return None, None

def create_working_onnx_model(model, approach):
    """
    Create ONNX model from working PyTorch model
    """
    print(f"\n🚀 CREATING WORKING ONNX MODEL")
    print("="*35)
    
    try:
        # Create raw output wrapper
        class WorkingRawModel(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x):
                outputs = self.backbone(x)
                
                # Handle different output formats
                if len(outputs) >= 4:
                    # Standard format: features, regression, classification, anchors
                    return outputs[1], outputs[2]  # regression, classification
                elif len(outputs) >= 2:
                    # Alternative format
                    return outputs[0], outputs[1]
                else:
                    raise ValueError(f"Unexpected output format: {len(outputs)} outputs")
        
        raw_model = WorkingRawModel(model)
        
        # Test ONNX export
        dummy_input = torch.randn(1, 3, 768, 768)
        
        # Test raw model first
        with torch.no_grad():
            raw_outputs = raw_model(dummy_input)
        
        print(f"✅ Raw model test successful")
        print(f"   Output shapes: {[out.shape for out in raw_outputs]}")
        
        # Export to ONNX
        output_filename = f"efficientdet_abhid_4classes_{approach['compound_coef']}.onnx"
        
        torch.onnx.export(
            raw_model,
            dummy_input,
            output_filename,
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
        
        print(f"✅ ONNX model created: {output_filename}")
        
        # Verify ONNX model
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_filename, providers=['CPUExecutionProvider'])
        onnx_outputs = session.run(None, {'input': dummy_input.numpy()})
        
        print(f"✅ ONNX verification successful")
        print(f"   ONNX output shapes: {[out.shape for out in onnx_outputs]}")
        
        return output_filename
        
    except Exception as e:
        print(f"❌ ONNX creation failed: {e}")
        return None

def main():
    """
    Main workflow for Abhid's specific model
    """
    print("🎯 ABHID'S 4-CLASS EFFICIENTDET CONVERSION FIX")
    print("="*50)
    
    # Step 1: Load exact training configuration
    config = load_exact_training_config()
    
    # Step 2: Try different approaches to find working model
    working_model, working_approach = test_with_different_approaches(config)
    
    if working_model is None:
        print(f"\n❌ CRITICAL: No working model found")
        print(f"📋 This suggests a fundamental issue:")
        print(f"   1. Checkpoint may be corrupted")
        print(f"   2. PyTorch version incompatibility")
        print(f"   3. Custom model modifications")
        
        print(f"\n🔧 MANUAL STEPS TO TRY:")
        print(f"   1. Check PyTorch version: python -c 'import torch; print(torch.__version__)'")
        print(f"   2. Try in original training environment")
        print(f"   3. Check if model loads in training script")
        return
    
    # Step 3: Create ONNX model
    onnx_filename = create_working_onnx_model(working_model, working_approach)
    
    if onnx_filename:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Working model: {working_approach['name']}")
        print(f"✅ ONNX file: {onnx_filename}")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Test ONNX model:")
        print(f"      python raw_onnx_inference.py --model {onnx_filename} --image test.jpg")
        print(f"   2. Run evaluation:")
        print(f"      python onnx_coco_eval.py --model {onnx_filename} ...")
        print(f"   3. Compare with original 72% mAP")
        
    else:
        print(f"\n❌ ONNX creation failed")
        print(f"But we found a working PyTorch model: {working_approach['name']}")
        print(f"Try manual ONNX export or use the PyTorch model directly")

if __name__ == '__main__':
    main()