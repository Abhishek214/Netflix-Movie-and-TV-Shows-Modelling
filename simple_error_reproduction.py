#!/usr/bin/env python3
"""
Super simple script to reproduce the exact strides error
Run this to see exactly where it fails
"""

import torch
import sys
import traceback

# Add your path
sys.path.append('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master')

print("🚨 REPRODUCING STRIDES ERROR")
print("="*40)

try:
    print("Step 1: Importing model...")
    from backbone import EfficientDetBackbone
    print("✅ Import successful")
    
    print("Step 2: Creating model...")
    model = EfficientDetBackbone(num_classes=4, compound_coef=2)
    print("✅ Model creation successful")
    
    print("Step 3: Loading checkpoint...")
    checkpoint = torch.load('/opt/disk1/app/Abhishek/Yet-Another-EfficientDet-Pytorch-master/logs/abhid/efficientdet-d2_49_8700.pth', map_location='cpu')
    print("✅ Checkpoint loaded")
    
    print("Step 4: Loading state dict...")
    model.load_state_dict(checkpoint['model'], strict=False)
    print("✅ State dict loaded")
    
    print("Step 5: Setting eval mode...")
    model.eval()
    print("✅ Eval mode set")
    
    print("Step 6: Creating test input...")
    test_input = torch.randn(1, 3, 768, 768)
    print("✅ Test input created")
    
    print("Step 7: Running forward pass (this should trigger strides error)...")
    with torch.no_grad():
        outputs = model(test_input)
    
    print("✅ NO ERROR! Forward pass successful!")
    print(f"Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
    
    print("\n🎉 SUCCESS! No strides error found!")
    print("You can now export to ONNX safely.")
    
except Exception as e:
    print(f"\n🚨 ERROR CAUGHT!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
    print("\n📍 FULL TRACEBACK:")
    traceback.print_exc()
    
    print(f"\n🔍 ERROR ANALYSIS:")
    
    error_str = str(e).lower()
    
    if "strides" in error_str:
        print("✅ This is indeed a strides error!")
        
        if "incorrect size" in error_str:
            print("🔍 Type: Size mismatch in strides")
            print("💡 Cause: Convolution layer expects different input dimensions")
        
        if "attribute" in error_str:
            print("🔍 Type: Attribute error on strides")
            print("💡 Cause: Layer trying to access strides property incorrectly")
        
        print(f"\n🔧 IMMEDIATE FIXES TO TRY:")
        print(f"1. Change num_classes to different values: 1, 80, 90")
        print(f"2. Change compound_coef to: 0, 1")
        print(f"3. Use smaller input size: 512x512 instead of 768x768")
        print(f"4. Check if checkpoint is corrupted")
        
    elif "size mismatch" in error_str or "shape" in error_str:
        print("🔍 This is a shape/size mismatch error")
        print("💡 Likely cause: Model architecture doesn't match checkpoint")
        
    elif "cuda" in error_str or "device" in error_str:
        print("🔍 This is a device/CUDA error")
        print("💡 Try running on CPU only")
        
    else:
        print("🔍 Unknown error type")
        print("💡 Check the full traceback above for clues")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Run the comprehensive debugger: python deep_strides_debugger.py")
    print(f"2. Run the targeted fix: python targeted_strides_fix.py")
    print(f"3. Try different model parameters based on the error type above")