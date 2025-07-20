#!/usr/bin/env python3
"""
Deep debugging script to find the exact root cause of strides errors
This will trace through every layer and identify the mismatch
"""

import torch
import torch.nn as nn
import sys
import traceback
import json

# Add your project path
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

class StridesErrorDetective:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = None
        self.state_dict = None
        
    def load_checkpoint(self):
        """Load and analyze checkpoint structure"""
        print("🔍 STEP 1: ANALYZING CHECKPOINT STRUCTURE")
        print("="*50)
        
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            print(f"✅ Checkpoint loaded from: {self.checkpoint_path}")
            
            # Extract state dict
            if 'model' in self.checkpoint:
                self.state_dict = self.checkpoint['model']
                print("✅ Found 'model' key in checkpoint")
            elif 'state_dict' in self.checkpoint:
                self.state_dict = self.checkpoint['state_dict']
                print("✅ Found 'state_dict' key in checkpoint")
            else:
                self.state_dict = self.checkpoint
                print("✅ Using checkpoint directly as state_dict")
            
            # Analyze checkpoint metadata
            print(f"\n📋 Checkpoint analysis:")
            print(f"   Total keys: {len(self.state_dict)}")
            
            # Check for additional metadata
            for key in self.checkpoint.keys():
                if key != 'model' and key != 'state_dict':
                    print(f"   Extra key '{key}': {type(self.checkpoint[key])}")
                    if key in ['epoch', 'step', 'compound_coef', 'num_classes']:
                        print(f"      Value: {self.checkpoint[key]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            traceback.print_exc()
            return False
    
    def analyze_key_patterns(self):
        """Analyze key patterns to understand model structure"""
        print(f"\n🔍 STEP 2: ANALYZING KEY PATTERNS")
        print("="*40)
        
        if not self.state_dict:
            print("❌ No state dict available")
            return
        
        # Group keys by module
        key_groups = {}
        for key in self.state_dict.keys():
            parts = key.split('.')
            if len(parts) >= 2:
                module_name = parts[0]
                if module_name not in key_groups:
                    key_groups[module_name] = []
                key_groups[module_name].append(key)
        
        print(f"📋 Model modules found:")
        for module_name, keys in key_groups.items():
            print(f"   {module_name}: {len(keys)} parameters")
            
            # Show sample keys for each module
            sample_keys = keys[:3]
            for sample_key in sample_keys:
                tensor = self.state_dict[sample_key]
                print(f"      {sample_key}: {tensor.shape}")
        
        # Look for backbone-specific patterns
        backbone_keys = [k for k in self.state_dict.keys() if 'backbone' in k or 'efficientnet' in k]
        classifier_keys = [k for k in self.state_dict.keys() if 'classifier' in k or 'header' in k]
        regressor_keys = [k for k in self.state_dict.keys() if 'regressor' in k or 'box' in k]
        
        print(f"\n📋 Component analysis:")
        print(f"   Backbone keys: {len(backbone_keys)}")
        print(f"   Classifier keys: {len(classifier_keys)}")
        print(f"   Regressor keys: {len(regressor_keys)}")
        
        return key_groups
    
    def test_model_creation_step_by_step(self):
        """Test model creation with different configurations step by step"""
        print(f"\n🔍 STEP 3: TESTING MODEL CREATION CONFIGURATIONS")
        print("="*55)
        
        # Import model classes
        try:
            from backbone import EfficientDetBackbone
            from efficientdet.model import EfficientDet
            print("✅ Successfully imported model classes")
        except Exception as e:
            print(f"❌ Error importing model classes: {e}")
            return None
        
        # Test configurations systematically
        test_configs = [
            # Basic configurations
            {"name": "D2_4classes", "compound_coef": 2, "num_classes": 4},
            {"name": "D2_80classes", "compound_coef": 2, "num_classes": 80},
            {"name": "D1_4classes", "compound_coef": 1, "num_classes": 4},
            {"name": "D0_4classes", "compound_coef": 0, "num_classes": 4},
            
            # Different class counts
            {"name": "D2_1class", "compound_coef": 2, "num_classes": 1},
            {"name": "D2_10classes", "compound_coef": 2, "num_classes": 10},
            {"name": "D2_20classes", "compound_coef": 2, "num_classes": 20},
        ]
        
        successful_configs = []
        
        for config in test_configs:
            print(f"\n🧪 Testing: {config['name']}")
            
            try:
                # Create model
                model = EfficientDetBackbone(
                    num_classes=config['num_classes'],
                    compound_coef=config['compound_coef']
                )
                
                # Try to load state dict
                missing, unexpected = model.load_state_dict(self.state_dict, strict=False)
                
                print(f"   📊 Missing keys: {len(missing)}")
                print(f"   📊 Unexpected keys: {len(unexpected)}")
                
                # Test forward pass with small input first
                model.eval()
                test_input = torch.randn(1, 3, 256, 256)  # Smaller input first
                
                with torch.no_grad():
                    outputs = model(test_input)
                
                print(f"   ✅ SUCCESS with 256x256 input")
                
                # Try with full resolution
                test_input = torch.randn(1, 3, 768, 768)
                with torch.no_grad():
                    outputs = model(test_input)
                
                print(f"   ✅ SUCCESS with 768x768 input")
                print(f"   📊 Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
                
                successful_configs.append((config, model))
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
                
                # Detailed error analysis
                error_str = str(e).lower()
                if "strides" in error_str:
                    print(f"   🔍 STRIDES ERROR DETECTED")
                    self.analyze_strides_error(e, config)
                elif "size mismatch" in error_str:
                    print(f"   🔍 SIZE MISMATCH ERROR")
                elif "shape" in error_str:
                    print(f"   🔍 SHAPE ERROR")
        
        return successful_configs
    
    def analyze_strides_error(self, error, config):
        """Analyze the specific strides error"""
        print(f"      🔬 DETAILED STRIDES ERROR ANALYSIS:")
        
        error_message = str(error)
        print(f"      Error message: {error_message}")
        
        # Extract specific information from error
        if "attribute" in error_message.lower() and "strides" in error_message.lower():
            print(f"      🔍 This is an attribute access error on 'strides'")
            print(f"      🔍 Likely cause: Layer expects different input format")
        
        if "incorrect size" in error_message.lower():
            print(f"      🔍 This is a size mismatch in strides")
            print(f"      🔍 Likely cause: Convolution layer configuration mismatch")
        
        # Try to trace the exact layer causing the issue
        try:
            self.trace_forward_pass_error(config)
        except Exception as trace_error:
            print(f"      🔍 Could not trace exact layer: {trace_error}")
    
    def trace_forward_pass_error(self, config):
        """Trace exactly which layer is causing the strides error"""
        print(f"      🔬 TRACING FORWARD PASS:")
        
        from backbone import EfficientDetBackbone
        
        # Create model
        model = EfficientDetBackbone(
            num_classes=config['num_classes'],
            compound_coef=config['compound_coef']
        )
        
        # Load weights
        model.load_state_dict(self.state_dict, strict=False)
        model.eval()
        
        # Create hooks to trace forward pass
        def create_hook(name):
            def hook(module, input, output):
                try:
                    input_shapes = [inp.shape if hasattr(inp, 'shape') else type(inp) for inp in input]
                    output_shape = output.shape if hasattr(output, 'shape') else type(output)
                    print(f"         Layer {name}: {input_shapes} -> {output_shape}")
                except Exception as e:
                    print(f"         Layer {name}: Hook error - {e}")
            return hook
        
        # Register hooks on all modules
        hooks = []
        for name, module in model.named_modules():
            if len(name) > 0:  # Skip the root module
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # Run forward pass
        try:
            test_input = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                outputs = model(test_input)
        except Exception as e:
            print(f"      ❌ Forward pass failed at specific layer")
            print(f"      Error: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    def extract_model_metadata(self):
        """Extract any model metadata from checkpoint"""
        print(f"\n🔍 STEP 4: EXTRACTING MODEL METADATA")
        print("="*40)
        
        # Look for configuration in checkpoint
        metadata = {}
        
        for key in self.checkpoint.keys():
            if key not in ['model', 'state_dict']:
                metadata[key] = self.checkpoint[key]
                print(f"   {key}: {self.checkpoint[key]}")
        
        # Try to infer configuration from state dict shapes
        print(f"\n📊 Inferring configuration from weights:")
        
        # Look for classifier/regressor heads to determine number of classes
        classifier_keys = [k for k in self.state_dict.keys() if 'classifier' in k and 'weight' in k]
        regressor_keys = [k for k in self.state_dict.keys() if 'regressor' in k and 'weight' in k]
        
        for key in classifier_keys[:3]:  # Check first few
            weight_shape = self.state_dict[key].shape
            print(f"   Classifier {key}: {weight_shape}")
            
            # Try to infer number of classes
            if len(weight_shape) >= 2:
                possible_classes = weight_shape[0] // 9  # 9 anchors per location
                if possible_classes > 0:
                    print(f"      Possible classes: {possible_classes}")
        
        return metadata
    
    def generate_working_model_script(self, successful_configs):
        """Generate a working model script based on successful configurations"""
        print(f"\n🔍 STEP 5: GENERATING WORKING MODEL SCRIPT")
        print("="*45)
        
        if not successful_configs:
            print("❌ No successful configurations found")
            return
        
        # Use the first successful configuration
        config, model = successful_configs[0]
        
        print(f"✅ Using configuration: {config['name']}")
        
        script_content = f"""#!/usr/bin/env python3
'''
WORKING ONNX CONVERSION SCRIPT
Generated automatically after successful architecture matching
'''

import torch
import sys
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

from backbone import EfficientDetBackbone

def create_working_model():
    # WORKING CONFIGURATION (tested successfully)
    model = EfficientDetBackbone(
        num_classes={config['num_classes']},
        compound_coef={config['compound_coef']},
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # Load checkpoint
    checkpoint = torch.load('{self.checkpoint_path}', map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    
    # Load weights (non-strict to handle any mismatches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights: {{len(missing)}} missing, {{len(unexpected)}} unexpected")
    
    model.eval()
    return model

def export_to_onnx():
    model = create_working_model()
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 768, 768)
    with torch.no_grad():
        outputs = model(dummy_input)
    print("✅ Forward pass successful!")
    
    # Create raw output wrapper
    class RawModel(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        
        def forward(self, x):
            features, regression, classification, anchors = self.backbone(x)
            return regression, classification
    
    raw_model = RawModel(model)
    
    # Export to ONNX
    torch.onnx.export(
        raw_model, dummy_input, 'efficientdet_WORKING.onnx',
        export_params=True, opset_version=11, do_constant_folding=False,
        input_names=['input'], output_names=['regression', 'classification']
    )
    
    print("✅ WORKING ONNX model created: efficientdet_WORKING.onnx")

if __name__ == '__main__':
    export_to_onnx()
"""
        
        # Save the script
        with open('working_model_script.py', 'w') as f:
            f.write(script_content)
        
        print(f"✅ Working script saved as: working_model_script.py")
        print(f"📋 To use it, run: python working_model_script.py")
    
    def run_complete_analysis(self):
        """Run the complete debugging analysis"""
        print("🕵️ STRIDES ERROR ROOT CAUSE ANALYSIS")
        print("="*60)
        
        # Step 1: Load checkpoint
        if not self.load_checkpoint():
            return
        
        # Step 2: Analyze key patterns
        key_groups = self.analyze_key_patterns()
        
        # Step 3: Test configurations
        successful_configs = self.test_model_creation_step_by_step()
        
        # Step 4: Extract metadata
        metadata = self.extract_model_metadata()
        
        # Step 5: Generate working script
        if successful_configs:
            self.generate_working_model_script(successful_configs)
        
        # Final summary
        print(f"\n🎯 ANALYSIS SUMMARY")
        print("="*30)
        
        if successful_configs:
            print(f"✅ Found {len(successful_configs)} working configurations:")
            for config, _ in successful_configs:
                print(f"   - {config['name']}: compound_coef={config['compound_coef']}, num_classes={config['num_classes']}")
            
            print(f"\n📋 SOLUTION:")
            print(f"   1. Use configuration: {successful_configs[0][0]['name']}")
            print(f"   2. Run: python working_model_script.py")
            print(f"   3. Test: python raw_onnx_inference.py --model efficientdet_WORKING.onnx --image test.jpg")
        else:
            print(f"❌ No working configurations found")
            print(f"📋 POSSIBLE CAUSES:")
            print(f"   1. Checkpoint is corrupted")
            print(f"   2. Model architecture has changed since training")
            print(f"   3. Missing dependencies or wrong versions")
            print(f"   4. Custom modifications in the model code")
            
            print(f"\n📋 MANUAL DEBUGGING STEPS:")
            print(f"   1. Check if you can load the model in the original training environment")
            print(f"   2. Verify PyTorch version matches training")
            print(f"   3. Check for any custom model modifications")
            print(f"   4. Try loading just the backbone without the full EfficientDetBackbone")

def main():
    checkpoint_path = '/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth'
    
    detective = StridesErrorDetective(checkpoint_path)
    detective.run_complete_analysis()

if __name__ == '__main__':
    main()