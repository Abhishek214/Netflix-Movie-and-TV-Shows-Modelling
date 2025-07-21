"""
Modified convert_onnx.py with post-processing
Place this in convert/ directory and modify based on your repository structure
"""

import numpy as np
import torch
from torch import nn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
import torch.nn.functional as F


class EfficientDetONNX(nn.Module):
    """EfficientDet model with post-processing for ONNX export"""
    
    def __init__(self, backbone, threshold=0.05, iou_threshold=0.5, max_detections=100):
        super(EfficientDetONNX, self).__init__()
        self.backbone = backbone
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        
        # Initialize bbox transforms
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
    
    def nms(self, dets, thresh):
        """Simple NMS implementation"""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]

        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1 + 1).clamp(min=0)
            h = (yy2 - yy1 + 1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= thresh).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        
        return torch.LongTensor(keep)

    def forward(self, x):
        regression, classification, anchors = self.backbone(x)
        
        # Apply transforms
        transformed_anchors = self.regressBoxes(anchors, regression[0])
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        
        # Apply sigmoid to get probabilities
        scores = torch.sigmoid(classification[0])
        
        # Get the maximum class score and class index for each anchor
        scores_max, classes = torch.max(scores, dim=1)
        
        # Filter by confidence threshold
        anchors_nms_idx = torch.where(scores_max > self.threshold)[0]
        
        if anchors_nms_idx.shape[0] == 0:
            # Return empty tensor if no detections
            return torch.zeros((0, 7), dtype=torch.float32)
        
        nms_scores, nms_class = scores[anchors_nms_idx].max(dim=1)
        nms_boxes = transformed_anchors[anchors_nms_idx]
        
        # Prepare for NMS
        dets = torch.cat([
            nms_boxes,
            nms_scores.unsqueeze(1)
        ], dim=1)
        
        # Apply NMS
        keep = self.nms(dets, self.iou_threshold)
        
        if len(keep) == 0:
            return torch.zeros((0, 7), dtype=torch.float32)
        
        # Limit detections
        if len(keep) > self.max_detections:
            keep = keep[:self.max_detections]
        
        # Final detections
        final_boxes = nms_boxes[keep]
        final_scores = nms_scores[keep]
        final_classes = nms_class[keep]
        
        # Create batch IDs (always 0 for single batch)
        batch_ids = torch.zeros((len(keep), 1))
        
        # Format: [batch_id, x1, y1, x2, y2, confidence, class_id]
        detections = torch.cat([
            batch_ids,
            final_boxes,
            final_scores.unsqueeze(1),
            final_classes.unsqueeze(1).float()
        ], dim=1)
        
        return detections


def main():
    # Configuration
    compound_coef = 2  # Change this for your model
    img_size = 768     # 512 for D0, 640 for D1, 768 for D2, etc.
    
    # Model weights path - UPDATE THIS
    weights_path = 'path/to/your/trained_model.pth'
    
    # Output path
    output_path = f'efficientdet_d{compound_coef}_with_postproc.onnx'
    
    # Load model
    model = EfficientDetBackbone(
        num_classes=90,  # COCO classes, change if different
        compound_coef=compound_coef,
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Wrap with post-processing
    onnx_model = EfficientDetONNX(
        backbone=model,
        threshold=0.05,
        iou_threshold=0.5,
        max_detections=100
    )
    onnx_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Test model
    print("Testing model...")
    with torch.no_grad():
        output = onnx_model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Sample output: {output[:5] if len(output) > 0 else 'No detections'}")
    
    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['detections'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'detections': {0: 'num_detections'}
        },
        opset_version=11,
        export_params=True,
        do_constant_folding=True
    )
    
    print(f"Model exported successfully!")
    print(f"Output format: [batch_id, x1, y1, x2, y2, confidence, class_id]")


if __name__ == '__main__':
    main()
