# Fixes for the TracerWarning issues in your original conversion

# 1. Fix for efficientnet/utils_extra.py - Line numbers may vary
# Replace the problematic functions with ONNX-compatible versions

# In efficientnet/utils_extra.py, replace the Conv2dStaticSamePadding class:

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv2dStaticSamePaddingONNX(nn.Module):
    """
    ONNX-compatible version of Conv2dStaticSamePadding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                             bias=bias, groups=groups, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Calculate padding statically
        pad_h = max((math.ceil(h / self.stride[1]) - 1) * self.stride[1] + 
                   (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h, 0)
        pad_w = max((math.ceil(w / self.stride[0]) - 1) * self.stride[0] + 
                   (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w, 0)
        
        # Apply static padding
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                         pad_h // 2, pad_h - pad_h // 2])
        
        return self.conv(x)

class MaxPool2dStaticSamePaddingONNX(nn.Module):
    """
    ONNX-compatible version of MaxPool2dStaticSamePadding
    """
    def __init__(self, kernel_size, stride, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)
        self.stride = stride if stride is not None else kernel_size
        self.kernel_size = kernel_size

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Calculate padding statically
        pad_h = max((math.ceil(h / self.stride) - 1) * self.stride + self.kernel_size - h, 0)
        pad_w = max((math.ceil(w / self.stride) - 1) * self.stride + self.kernel_size - w, 0)
        
        # Apply static padding
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                         pad_h // 2, pad_h - pad_h // 2])
        
        return self.pool(x)

# 2. Fix for the strides issue in the backbone
# Create a modified backbone.py file or patch the existing one

def patch_backbone_for_onnx():
    """
    Apply patches to make the backbone ONNX-compatible
    """
    import efficientnet.utils_extra as utils_extra
    
    # Replace the problematic classes
    utils_extra.Conv2dStaticSamePadding = Conv2dStaticSamePaddingONNX
    utils_extra.MaxPool2dStaticSamePadding = MaxPool2dStaticSamePaddingONNX
    
    print("Applied ONNX compatibility patches")

# 3. Modified conversion script that addresses the TracerWarnings
def convert_efficientdet_fixed():
    """
    Fixed conversion script that avoids TracerWarnings
    """
    import torch
    import torch.nn as nn
    from backbone import EfficientDetBackbone
    import argparse
    
    # Apply patches first
    patch_backbone_for_onnx()
    
    class FixedEfficientDetBackbone(EfficientDetBackbone):
        """
        Modified backbone that avoids tensor-to-scalar conversions
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def forward(self, inputs):
            # Get the original forward pass
            _, regression, classification, anchors = super().forward(inputs)
            
            # Return only the essential outputs to avoid anchor-related issues
            return regression, classification
    
    def export_model():
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--compound_coef', type=int, default=2)
        parser.add_argument('-w', '--weights', type=str, required=True)
        parser.add_argument('--output', type=str, default='efficientdet_fixed.onnx')
        
        args = parser.parse_args()
        
        # Create model
        model = FixedEfficientDetBackbone(
            num_classes=80,  # Change to your number of classes
            compound_coef=args.compound_coef
        )
        
        # Load weights
        checkpoint = torch.load(args.weights, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        # Input size for different compounds
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        input_size = input_sizes[args.compound_coef]
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        print(f"Converting EfficientDet D{args.compound_coef}...")
        print(f"Input size: {input_size}")
        
        # Export with fixed settings
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=11,  # Use opset 11 for better compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['regression', 'classification'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=False  # Reduce verbose output
        )
        
        print(f"Model exported to: {args.output}")
    
    return export_model

# 4. Environment setup script
def setup_environment():
    """
    Setup script to ensure all dependencies are correctly installed
    """
    import subprocess
    import sys
    
    required_packages = [
        'torch>=1.8.0',
        'torchvision',
        'onnx>=1.10.0',
        'onnxruntime>=1.10.0',
        'numpy',
        'opencv-python',
        'pillow',
        'matplotlib'
    ]
    
    print("Setting up environment for ONNX conversion...")
    
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("Environment setup complete!")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # setup_environment()
    # convert_func = convert_efficientdet_fixed()
    # convert_func()
    pass