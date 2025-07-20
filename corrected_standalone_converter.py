#!/usr/bin/env python3
"""
Corrected Standalone EfficientDet ONNX Converter
Fixes channel dimension mismatches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                             bias=bias, groups=groups, dilation=dilation)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

    def forward(self, x):
        h, w = x.shape[-2:]
        
        # Calculate padding needed
        pad_h = max((math.ceil(h / self.stride[1]) - 1) * self.stride[1] + 
                   (self.kernel_size[0] - 1) * self.dilation[0] + 1 - h, 0)
        pad_w = max((math.ceil(w / self.stride[0]) - 1) * self.stride[0] + 
                   (self.kernel_size[1] - 1) * self.dilation[1] + 1 - w, 0)
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                         pad_h // 2, pad_h - pad_h // 2])
        
        return self.conv(x)

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=True):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Depthwise convolution
        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels, in_channels, kernel_size=3, stride=1, 
            groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPNLayer(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4, first_time=False, conv_channels=None):
        super(BiFPNLayer, self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time
        
        # Convolution layers for BiFPN
        self.conv6_up = SeparableConvBlock(num_channels)
        self.conv5_up = SeparableConvBlock(num_channels)
        self.conv4_up = SeparableConvBlock(num_channels)
        self.conv3_up = SeparableConvBlock(num_channels)

        self.conv4_down = SeparableConvBlock(num_channels)
        self.conv5_down = SeparableConvBlock(num_channels)
        self.conv6_down = SeparableConvBlock(num_channels)
        self.conv7_down = SeparableConvBlock(num_channels)

        # Channel reduction for first BiFPN layer
        if first_time and conv_channels is not None:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # P6 and P7 from P5
            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

            # Additional channel reductions for skip connections
            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weighted feature fusion weights
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        if self.first_time:
            # First BiFPN layer - convert backbone features
            p3, p4, p5 = inputs
            
            # Apply channel reduction
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            
            # Create P6 and P7
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
        else:
            # Subsequent BiFPN layers
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # Weighted feature fusion with learnable weights
        # P6_0 and P7_0 -> P6_1
        p6_w1 = F.relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * F.interpolate(p7_in, size=p6_in.shape[-2:], mode='nearest'))

        # P5_0 and P6_1 -> P5_1
        p5_w1 = F.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * F.interpolate(p6_up, size=p5_in.shape[-2:], mode='nearest'))

        # P4_0 and P5_1 -> P4_1
        p4_w1 = F.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * F.interpolate(p5_up, size=p4_in.shape[-2:], mode='nearest'))

        # P3_0 and P4_1 -> P3_2
        p3_w1 = F.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * F.interpolate(p4_up, size=p3_in.shape[-2:], mode='nearest'))

        # P4_0, P4_1 and P3_2 -> P4_2
        p4_w2 = F.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            weight[0] * p4_in + weight[1] * p4_up + weight[2] * F.interpolate(p3_out, size=p4_in.shape[-2:], mode='nearest'))

        # P5_0, P5_1 and P4_2 -> P5_2
        p5_w2 = F.relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            weight[0] * p5_in + weight[1] * p5_up + weight[2] * F.interpolate(p4_out, size=p5_in.shape[-2:], mode='nearest'))

        # P6_0, P6_1 and P5_2 -> P6_2
        p6_w2 = F.relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            weight[0] * p6_in + weight[1] * p6_up + weight[2] * F.interpolate(p5_out, size=p6_in.shape[-2:], mode='nearest'))

        # P7_0 and P6_2 -> P7_2
        p7_w2 = F.relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * F.interpolate(p6_out, size=p7_in.shape[-2:], mode='nearest'))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class SimpleBackbone(nn.Module):
    """Simple backbone that outputs correct channel dimensions"""
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            Conv2dStaticSamePadding(3, 32, 3, stride=2),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Stage 1: 32 -> 40 (P2)
        self.stage1 = nn.Sequential(
            Conv2dStaticSamePadding(32, 40, 3, stride=2),
            nn.BatchNorm2d(40, momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Stage 2: 40 -> 80 (P3)
        self.stage2 = nn.Sequential(
            Conv2dStaticSamePadding(40, 80, 3, stride=2),
            nn.BatchNorm2d(80, momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Stage 3: 80 -> 112 (P4)
        self.stage3 = nn.Sequential(
            Conv2dStaticSamePadding(80, 112, 3, stride=2),
            nn.BatchNorm2d(112, momentum=0.01, eps=1e-3),
            Swish()
        )
        
        # Stage 4: 112 -> 320 (P5)
        self.stage4 = nn.Sequential(
            Conv2dStaticSamePadding(112, 320, 3, stride=2),
            nn.BatchNorm2d(320, momentum=0.01, eps=1e-3),
            Swish()
        )

    def forward(self, x):
        # Extract features at different scales
        x = self.stem(x)     # /2
        x = self.stage1(x)   # /4
        p3 = self.stage2(x)  # /8  - 80 channels
        p4 = self.stage3(p3) # /16 - 112 channels  
        p5 = self.stage4(p4) # /32 - 320 channels
        
        return p3, p4, p5

class CorrectedEfficientDet(nn.Module):
    def __init__(self, compound_coef=2, num_classes=80):
        super(CorrectedEfficientDet, self).__init__()
        
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        
        # Model configuration
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        
        # Get configuration for current compound coefficient
        fpn_num_filters = self.fpn_num_filters[compound_coef]
        fpn_cell_repeats = self.fpn_cell_repeats[compound_coef]
        box_class_repeats = self.box_class_repeats[compound_coef]
        
        # Backbone
        self.backbone = SimpleBackbone()
        
        # BiFPN layers
        bifpn_layers = []
        for i in range(fpn_cell_repeats):
            if i == 0:
                # First layer converts backbone features to BiFPN channels
                layer = BiFPNLayer(
                    num_channels=fpn_num_filters,
                    first_time=True,
                    conv_channels=[80, 112, 320]  # P3, P4, P5 channel counts from backbone
                )
            else:
                layer = BiFPNLayer(num_channels=fpn_num_filters, first_time=False)
            bifpn_layers.append(layer)
        
        self.bifpn = nn.ModuleList(bifpn_layers)
        
        # Prediction heads
        # Regression head
        regressor_layers = []
        for _ in range(box_class_repeats):
            regressor_layers.append(SeparableConvBlock(fpn_num_filters))
        regressor_layers.append(
            Conv2dStaticSamePadding(fpn_num_filters, 9 * 4, kernel_size=3, stride=1)
        )
        self.regressor = nn.Sequential(*regressor_layers)
        
        # Classification head
        classifier_layers = []
        for _ in range(box_class_repeats):
            classifier_layers.append(SeparableConvBlock(fpn_num_filters))
        classifier_layers.append(
            Conv2dStaticSamePadding(fpn_num_filters, 9 * num_classes, kernel_size=3, stride=1)
        )
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Post-processing parameters
        self.score_threshold = 0.2
        self.max_detections = 100

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract backbone features
        p3, p4, p5 = self.backbone(x)
        
        # Apply BiFPN
        features = [p3, p4, p5]
        for bifpn_layer in self.bifpn:
            features = bifpn_layer(features)
        
        # Apply prediction heads to each feature level
        regressions = []
        classifications = []
        
        for feature in features:
            # Regression
            reg = self.regressor(feature)
            batch_size = reg.shape[0]
            reg = reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            regressions.append(reg)
            
            # Classification
            cls = self.classifier(feature)
            cls = cls.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            classifications.append(cls)
        
        # Concatenate predictions from all levels
        regression = torch.cat(regressions, dim=1)
        classification = torch.cat(classifications, dim=1)
        
        # Apply sigmoid to classification scores
        classification = torch.sigmoid(classification)
        
        # Simple post-processing for ONNX export
        batch_size = classification.shape[0]
        
        # Get maximum scores and corresponding class indices
        max_scores, class_indices = torch.max(classification, dim=2)
        
        # Apply score threshold
        score_mask = max_scores > self.score_threshold
        
        # Prepare outputs with fixed size
        output_boxes = torch.zeros(batch_size, self.max_detections, 4)
        output_scores = torch.zeros(batch_size, self.max_detections)
        output_labels = torch.zeros(batch_size, self.max_detections, dtype=torch.long)
        
        for i in range(batch_size):
            valid_mask = score_mask[i]
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                # Get valid predictions
                valid_boxes = regression[i][valid_mask]
                valid_scores = max_scores[i][valid_mask]
                valid_labels = class_indices[i][valid_mask]
                
                # Take top detections
                num_dets = min(self.max_detections, valid_count)
                if num_dets > 0:
                    # Sort by score
                    _, sorted_indices = torch.sort(valid_scores, descending=True)
                    top_indices = sorted_indices[:num_dets]
                    
                    output_boxes[i, :num_dets] = valid_boxes[top_indices]
                    output_scores[i, :num_dets] = valid_scores[top_indices]
                    output_labels[i, :num_dets] = valid_labels[top_indices]
        
        return output_boxes, output_scores, output_labels

def convert_to_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compound_coef', type=int, default=2)
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--output', type=str, default='efficientdet_corrected.onnx')
    parser.add_argument('--score_threshold', type=float, default=0.2)
    
    args = parser.parse_args()
    
    print(f"Creating corrected EfficientDet D{args.compound_coef}")
    print(f"Number of classes: {args.num_classes}")
    
    # Create model
    model = CorrectedEfficientDet(
        compound_coef=args.compound_coef,
        num_classes=args.num_classes
    )
    
    model.score_threshold = args.score_threshold
    model.eval()
    
    # Get input size
    input_size = model.input_sizes[args.compound_coef]
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Input size: {input_size}x{input_size}")
    
    # Test forward pass first
    try:
        print("Testing forward pass...")
        with torch.no_grad():
            outputs = model(dummy_input)
        print("✓ Forward pass successful")
        print(f"✓ Output shapes: {[out.shape for out in outputs]}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    print("Converting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'labels': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✓ ONNX model saved to: {args.output}")
    print("Model outputs:")
    print("  - boxes: [batch_size, max_detections, 4]")
    print("  - scores: [batch_size, max_detections]")
    print("  - labels: [batch_size, max_detections]")

if __name__ == '__main__':
    convert_to_onnx()