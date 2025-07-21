import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Tuple
import math

# Import your EfficientDet components
from efficientdet.model import EfficientNet, BiFPN, Regressor, Classifier, EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display


class EfficientDetWithPostProcessing(nn.Module):
    """
    EfficientDet model with full post-processing for ONNX export
    """
    def __init__(self, backbone, compound_coef=2, num_classes=90, 
                 ratios=None, scales=None, threshold=0.05, iou_threshold=0.5):
        super(EfficientDetWithPostProcessing, self).__init__()
        
        self.backbone = backbone
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        
        # Anchor parameters
        if ratios is None:
            ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        if scales is None:
            scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            
        self.ratios = ratios
        self.scales = scales
        
        # Image size for this compound coefficient
        img_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.img_size = img_sizes[compound_coef]
        
        # Initialize transforms
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        # Generate anchors
        self.anchors = self._generate_anchors()
        
    def _generate_anchors(self):
        """Generate anchor boxes for all pyramid levels"""
        anchor_scale = 4.
        pyramid_levels = [3, 4, 5, 6, 7]
        strides = [2 ** x for x in pyramid_levels]
        sizes = [anchor_scale * (2 ** x) for x in pyramid_levels]
        
        anchors = []
        for stride, size in zip(strides, sizes):
            anchors.append(self._generate_level_anchors(stride, size))
        
        return torch.cat(anchors, dim=0)
    
    def _generate_level_anchors(self, stride, size):
        """Generate anchors for a single pyramid level"""
        grid_size = self.img_size // stride
        
        # Create grid coordinates
        shift_x = torch.arange(0, grid_size) * stride
        shift_y = torch.arange(0, grid_size) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        shifts = torch.stack([shift_x.flatten(), shift_y.flatten(), 
                            shift_x.flatten(), shift_y.flatten()], dim=1).float()
        
        # Generate base anchors
        base_anchors = []
        for ratio in self.ratios:
            for scale in self.scales:
                w = size * scale * math.sqrt(ratio[0] / ratio[1])
                h = size * scale * math.sqrt(ratio[1] / ratio[0])
                
                base_anchor = torch.tensor([-w/2, -h/2, w/2, h/2]).float()
                base_anchors.append(base_anchor)
        
        base_anchors = torch.stack(base_anchors, dim=0)
        
        # Apply shifts to base anchors
        num_anchors = len(base_anchors)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        
        return all_anchors
    
    def _apply_nms(self, boxes, scores, iou_threshold=0.5, max_detections=100):
        """Apply Non-Maximum Suppression"""
        # Convert to torchvision format
        keep_indices = []
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for class_id in range(self.num_classes):
            class_scores = scores[:, class_id]
            valid_mask = class_scores > self.threshold
            
            if not valid_mask.any():
                continue
                
            class_boxes = boxes[valid_mask]
            class_scores = class_scores[valid_mask]
            
            # Apply NMS
            try:
                keep = torch.ops.torchvision.nms(class_boxes, class_scores, iou_threshold)
            except:
                # Fallback NMS implementation
                keep = self._nms_fallback(class_boxes, class_scores, iou_threshold)
            
            if len(keep) > 0:
                final_boxes.append(class_boxes[keep])
                final_scores.append(class_scores[keep])
                final_labels.append(torch.full((len(keep),), class_id, dtype=torch.int64))
        
        if len(final_boxes) == 0:
            # Return empty tensors with correct shape
            return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64)
        
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        
        # Limit to max detections
        if len(final_scores) > max_detections:
            top_indices = torch.topk(final_scores, max_detections)[1]
            final_boxes = final_boxes[top_indices]
            final_scores = final_scores[top_indices]
            final_labels = final_labels[top_indices]
        
        return final_boxes, final_scores, final_labels
    
    def _nms_fallback(self, boxes, scores, iou_threshold):
        """Fallback NMS implementation"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.int64)
        
        # Compute IoU matrix
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort(descending=True)
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.int64)
    
    def forward(self, x):
        # Get raw predictions from backbone
        regression, classification, anchors = self.backbone(x)
        
        batch_size = x.shape[0]
        device = x.device
        
        # Move anchors to device
        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)
        
        # Decode boxes
        transformed_anchors = self.regressBoxes(self.anchors, regression[0])
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        
        # Apply sigmoid to classification scores
        classification = torch.sigmoid(classification[0])
        
        # Apply NMS and post-processing
        final_boxes, final_scores, final_labels = self._apply_nms(
            transformed_anchors, classification, self.iou_threshold
        )
        
        # Prepare output in detection format
        if len(final_boxes) > 0:
            # Format: [batch_id, x1, y1, x2, y2, confidence, class_id]
            batch_ids = torch.zeros(len(final_boxes), 1, device=device)
            detections = torch.cat([
                batch_ids,
                final_boxes,
                final_scores.unsqueeze(1),
                final_labels.unsqueeze(1).float()
            ], dim=1)
        else:
            # Empty detections
            detections = torch.zeros(0, 7, device=device)
        
        return detections


def export_efficientdet_with_postprocessing(model_path, output_path, compound_coef=2, 
                                           img_size=None, threshold=0.05, iou_threshold=0.5):
    """
    Export EfficientDet model with full post-processing to ONNX
    """
    
    if img_size is None:
        img_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        img_size = img_sizes[compound_coef]
    
    print(f"Exporting EfficientDet D{compound_coef} with post-processing...")
    print(f"Input size: {img_size}x{img_size}")
    print(f"Confidence threshold: {threshold}")
    print(f"IoU threshold: {iou_threshold}")
    
    # Load the original model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize backbone (adjust this based on your model structure)
    backbone = EfficientDetBackbone(
        num_classes=90,  # COCO classes
        compound_coef=compound_coef,
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        backbone.load_state_dict(checkpoint['model'])
    else:
        backbone.load_state_dict(checkpoint)
    
    # Create model with post-processing
    model_with_postproc = EfficientDetWithPostProcessing(
        backbone=backbone,
        compound_coef=compound_coef,
        num_classes=90,
        threshold=threshold,
        iou_threshold=iou_threshold
    )
    
    model_with_postproc.to(device)
    model_with_postproc.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Test the model
    print("Testing model with post-processing...")
    with torch.no_grad():
        output = model_with_postproc(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output format: [batch_id, x1, y1, x2, y2, confidence, class_id]")
    
    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model_with_postproc,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['detections'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'detections': {0: 'num_detections'}
        },
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ Model exported successfully to: {output_path}")
    print(f"✓ Output format: [batch_id, x1, y1, x2, y2, confidence, class_id]")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export EfficientDet with full post-processing')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--output_path', type=str, default='efficientdet_with_postproc.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--compound_coef', type=int, default=2,
                        help='EfficientDet compound coefficient (0-7)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (auto-detected if None)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    export_efficientdet_with_postprocessing(
        model_path=args.model_path,
        output_path=args.output_path,
        compound_coef=args.compound_coef,
        img_size=args.img_size,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold
    )
