#!/usr/bin/env python3
"""
Standalone EfficientDet ONNX Converter
This script is completely self-contained and doesn't require the original repository structure.
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

class MemoryEfficientSwish(nn.Module):
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

class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, kernel_size, stride, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)
        self.stride = stride if stride is not None else kernel_size
        self.kernel_size = kernel_size

    def forward(self, x):
        h, w = x.shape[-2:]
        
        pad_h = max((math.ceil(h / self.stride) - 1) * self.stride + self.kernel_size - h, 0)
        pad_w = max((math.ceil(w / self.stride) - 1) * self.stride + self.kernel_size - w, 0)
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                         pad_h // 2, pad_h - pad_h // 2])
        
        return self.pool(x)

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=True, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, 
                                                      groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        if first_time:
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

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.attention = attention

    def forward(self, inputs):
        if self.attention:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            # Weights for P6_0 and P7_0 to P6_1
            p6_w1 = F.relu(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_up = self.conv6_up(weight[0] * p6_in + weight[1] * F.interpolate(p7_in, size=p6_in.shape[-2:], mode='nearest'))

            # Weights for P5_0 and P6_1 to P5_1
            p5_w1 = F.relu(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_up = self.conv5_up(weight[0] * p5_in + weight[1] * F.interpolate(p6_up, size=p5_in.shape[-2:], mode='nearest'))

            # Weights for P4_0 and P5_1 to P4_1
            p4_w1 = F.relu(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_up = self.conv4_up(weight[0] * p4_in + weight[1] * F.interpolate(p5_up, size=p4_in.shape[-2:], mode='nearest'))

            # Weights for P3_0 and P4_1 to P3_2
            p3_w1 = F.relu(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.conv3_up(weight[0] * p3_in + weight[1] * F.interpolate(p4_up, size=p3_in.shape[-2:], mode='nearest'))

            # Weights for P4_0, P4_1 and P3_2 to P4_2
            p4_w2 = F.relu(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.conv4_down(
                weight[0] * p4_in + weight[1] * p4_up + weight[2] * F.interpolate(p3_out, size=p4_in.shape[-2:], mode='nearest'))

            # Weights for P5_0, P5_1 and P4_2 to P5_2
            p5_w2 = F.relu(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.conv5_down(
                weight[0] * p5_in + weight[1] * p5_up + weight[2] * F.interpolate(p4_out, size=p5_in.shape[-2:], mode='nearest'))

            # Weights for P6_0, P6_1 and P5_2 to P6_2
            p6_w2 = F.relu(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.conv6_down(
                weight[0] * p6_in + weight[1] * p6_up + weight[2] * F.interpolate(p5_out, size=p6_in.shape[-2:], mode='nearest'))

            # Weights for P7_0 and P6_2 to P7_2
            p7_w2 = F.relu(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.conv7_down(weight[0] * p7_in + weight[1] * F.interpolate(p6_out, size=p7_in.shape[-2:], mode='nearest'))

            return p3_out, p4_out, p5_out, p6_out, p7_out

class StandaloneEfficientDet(nn.Module):
    def __init__(self, compound_coef=2, num_classes=80, ratios=None, scales=None):
        super(StandaloneEfficientDet, self).__init__()
        
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        
        self.num_classes = num_classes
        
        # Simple feature extractor instead of full EfficientNet
        self.backbone = self._make_simple_backbone()
        
        # BiFPN
        fpn_num_filters = self.fpn_num_filters[compound_coef]
        fpn_cell_repeats = self.fpn_cell_repeats[compound_coef]
        
        self.bifpn = nn.Sequential(
            *[BiFPN(fpn_num_filters, [128, 256, 512], 
                   first_time=(i == 0), attention=True, onnx_export=True)
              for i in range(fpn_cell_repeats)]
        )
        
        # Prediction heads
        box_class_repeats = self.box_class_repeats[compound_coef]
        
        # Regression head
        self.regressor = nn.Sequential(
            *[SeparableConvBlock(fpn_num_filters, onnx_export=True) 
              for _ in range(box_class_repeats)],
            Conv2dStaticSamePadding(fpn_num_filters, 9 * 4, kernel_size=3, stride=1)  # 9 anchors, 4 coordinates
        )
        
        # Classification head  
        self.classifier = nn.Sequential(
            *[SeparableConvBlock(fpn_num_filters, onnx_export=True) 
              for _ in range(box_class_repeats)],
            Conv2dStaticSamePadding(fpn_num_filters, 9 * num_classes, kernel_size=3, stride=1)  # 9 anchors
        )
        
        # Generate anchors
        self.anchors = self._generate_anchors()
        
        # Thresholds for post-processing
        self.score_threshold = 0.2
        self.nms_threshold = 0.5
        
    def _make_simple_backbone(self):
        """Create a simple backbone for demonstration"""
        return nn.Sequential(
            # Initial convolution
            Conv2dStaticSamePadding(3, 32, 3, stride=2),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            Swish(),
            
            # Feature layers
            Conv2dStaticSamePadding(32, 64, 3, stride=2),
            nn.BatchNorm2d(64, momentum=0.01, eps=1e-3),
            Swish(),
            
            Conv2dStaticSamePadding(64, 128, 3, stride=2),
            nn.BatchNorm2d(128, momentum=0.01, eps=1e-3),
            Swish(),
            
            Conv2dStaticSamePadding(128, 256, 3, stride=2),
            nn.BatchNorm2d(256, momentum=0.01, eps=1e-3),
            Swish(),
            
            Conv2dStaticSamePadding(256, 512, 3, stride=2),
            nn.BatchNorm2d(512, momentum=0.01, eps=1e-3),
            Swish(),
        )
    
    def _generate_anchors(self):
        """Generate anchor boxes for all pyramid levels"""
        pyramid_levels = [3, 4, 5, 6, 7]
        strides = [2 ** x for x in pyramid_levels]
        scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        
        # For simplicity, we'll generate anchors for a fixed input size
        input_size = self.input_sizes[self.compound_coef]
        
        all_anchors = []
        for stride in strides:
            anchors_level = []
            for scale in scales:
                for ratio in ratios:
                    h = stride * scale * ratio[1]
                    w = stride * scale * ratio[0]
                    
                    # Generate grid
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(0, input_size, stride, dtype=torch.float32),
                        torch.arange(0, input_size, stride, dtype=torch.float32),
                        indexing='ij'
                    )
                    
                    # Create anchors
                    anchors = torch.stack([
                        grid_x - w/2,  # x1
                        grid_y - h/2,  # y1  
                        grid_x + w/2,  # x2
                        grid_y + h/2   # y2
                    ], dim=-1)
                    
                    anchors_level.append(anchors.reshape(-1, 4))
            
            if anchors_level:
                all_anchors.append(torch.cat(anchors_level, dim=0))
        
        if all_anchors:
            return torch.cat(all_anchors, dim=0)
        else:
            # Fallback: create dummy anchors
            return torch.zeros(1000, 4)
    
    def forward(self, x):
        # Extract features from backbone
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 8, 12, 16, 20]:  # Extract features at different scales
                features.append(x)
        
        # Ensure we have 5 feature maps
        if len(features) < 5:
            # Pad with the last feature
            while len(features) < 5:
                features.append(features[-1])
        
        # Apply BiFPN
        features = self.bifpn(features)
        
        # Apply prediction heads
        regressions = []
        classifications = []
        
        for feature in features:
            regression = self.regressor(feature)
            classification = self.classifier(feature)
            
            # Reshape outputs
            batch_size = regression.shape[0]
            regression = regression.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            classification = classification.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            
            regressions.append(regression)
            classifications.append(classification)
        
        # Concatenate all predictions
        regression = torch.cat(regressions, dim=1)
        classification = torch.cat(classifications, dim=1)
        
        # Apply sigmoid to classifications
        classification = torch.sigmoid(classification)
        
        # Simple post-processing
        batch_size = regression.shape[0]
        num_anchors = regression.shape[1]
        
        # For ONNX export, we'll do simplified processing
        # Get max scores and classes
        scores, labels = torch.max(classification, dim=2)
        
        # Simple thresholding
        score_mask = scores > self.score_threshold
        
        # Prepare outputs
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for i in range(batch_size):
            valid_mask = score_mask[i]
            if valid_mask.sum() > 0:
                valid_boxes = regression[i][valid_mask]
                valid_scores = scores[i][valid_mask] 
                valid_labels = labels[i][valid_mask]
                
                # Take top detections to have fixed output size
                num_detections = min(100, valid_boxes.shape[0])
                if num_detections > 0:
                    top_indices = torch.topk(valid_scores, num_detections)[1]
                    boxes_list.append(valid_boxes[top_indices])
                    scores_list.append(valid_scores[top_indices])
                    labels_list.append(valid_labels[top_indices])
                else:
                    boxes_list.append(torch.zeros(1, 4))
                    scores_list.append(torch.zeros(1))
                    labels_list.append(torch.zeros(1, dtype=torch.long))
            else:
                boxes_list.append(torch.zeros(1, 4))
                scores_list.append(torch.zeros(1))
                labels_list.append(torch.zeros(1, dtype=torch.long))
        
        # Stack outputs (pad to same size if needed)
        max_detections = max(boxes.shape[0] for boxes in boxes_list)
        
        final_boxes = torch.zeros(batch_size, max_detections, 4)
        final_scores = torch.zeros(batch_size, max_detections)
        final_labels = torch.zeros(batch_size, max_detections, dtype=torch.long)
        
        for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
            num_det = boxes.shape[0]
            final_boxes[i, :num_det] = boxes
            final_scores[i, :num_det] = scores
            final_labels[i, :num_det] = labels
        
        return final_boxes, final_scores, final_labels

def convert_to_onnx():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compound_coef', type=int, default=2)
    parser.add_argument('-w', '--weights', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--output', type=str, default='efficientdet_standalone.onnx')
    parser.add_argument('--score_threshold', type=float, default=0.2)
    parser.add_argument('--nms_threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"Creating standalone EfficientDet D{args.compound_coef}")
    
    # Create model
    model = StandaloneEfficientDet(
        compound_coef=args.compound_coef,
        num_classes=args.num_classes
    )
    
    # Set thresholds
    model.score_threshold = args.score_threshold
    model.nms_threshold = args.nms_threshold
    
    # Load weights if provided (this is optional since we have a simplified model)
    if args.weights:
        try:
            print(f"Attempting to load weights from: {args.weights}")
            checkpoint = torch.load(args.weights, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Try to load compatible weights
            model_dict = model.state_dict()
            compatible_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
            
            if compatible_dict:
                model.load_state_dict(compatible_dict, strict=False)
                print(f"Loaded {len(compatible_dict)} compatible layers")
            else:
                print("No compatible weights found, using random initialization")
                
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Using random initialization")
    
    model.eval()
    
    # Get input size
    input_size = model.input_sizes[args.compound_coef]
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Input size: {input_size}x{input_size}")
    print(f"Converting to ONNX...")
    
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
    print("  - boxes: [batch_size, num_detections, 4]")
    print("  - scores: [batch_size, num_detections]") 
    print("  - labels: [batch_size, num_detections]")

if __name__ == '__main__':
    convert_to_onnx()