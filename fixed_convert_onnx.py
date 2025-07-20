import torch
import torch.nn as nn
import numpy as np
from efficientdet.model import EfficientDet, Anchors, BBoxTransform, ClipBoxes
from backbone import EfficientDetBackbone
import argparse

class EfficientDetONNXModel(nn.Module):
    """
    Modified EfficientDet model for ONNX export with post-processing included
    """
    def __init__(self, compound_coef=2, num_classes=80, ratios=None, scales=None):
        super(EfficientDetONNXModel, self).__init__()
        
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        
        # Load the backbone
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = ratios or [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.num_scales = len(scales or [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        
        # Build the model
        self.backbone_net = EfficientDetBackbone(
            num_classes=num_classes,
            compound_coef=compound_coef,
            ratios=self.aspect_ratios,
            scales=scales or [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        )
        
        # Anchor generation
        self.anchors = Anchors(
            anchor_scale=self.anchor_scale[compound_coef],
            pyramid_levels=(torch.arange(self.pyramid_levels[compound_coef]) + 3).tolist(),
            **{'ratios': self.aspect_ratios, 'scales': scales or [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]}
        )
        
        # Post-processing
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        # Confidence threshold for filtering
        self.threshold = 0.2
        self.iou_threshold = 0.2
        
    def forward(self, x):
        # Get predictions from backbone
        features, regression, classification, anchors = self.backbone_net(x)
        
        # Apply sigmoid to classification
        classification = torch.sigmoid(classification)
        
        # Transform boxes
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)
        
        # Apply confidence threshold
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.threshold)[0, :, 0]
        
        if scores_over_thresh.sum() == 0:
            # No boxes to NMS, return empty tensors
            batch_size = x.shape[0]
            return (
                torch.zeros((batch_size, 1, 4), dtype=torch.float32),  # boxes
                torch.zeros((batch_size, 1), dtype=torch.float32),     # scores  
                torch.zeros((batch_size, 1), dtype=torch.int64)        # labels
            )
        
        # Filter boxes and scores
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        
        # Get final scores and labels
        anchors_nms_idx = self.nms_pytorch(transformed_anchors[0], classification[0])
        
        if anchors_nms_idx.shape[0] != 0:
            classes = classification[0, anchors_nms_idx, :]
            scores, labels = torch.max(classes, dim=1)
            boxes = transformed_anchors[0, anchors_nms_idx, :]
            
            # Add batch dimension back
            boxes = boxes.unsqueeze(0)
            scores = scores.unsqueeze(0) 
            labels = labels.unsqueeze(0)
        else:
            # No boxes after NMS
            batch_size = x.shape[0]
            boxes = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
            scores = torch.zeros((batch_size, 1), dtype=torch.float32)
            labels = torch.zeros((batch_size, 1), dtype=torch.int64)
        
        return boxes, scores, labels
    
    def nms_pytorch(self, boxes, scores):
        """
        PyTorch implementation of NMS for ONNX compatibility
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64)
        
        max_scores, labels = torch.max(scores, dim=1)
        
        # Sort by score
        sorted_indices = torch.argsort(max_scores, descending=True)
        
        keep = []
        while sorted_indices.numel() > 0:
            if sorted_indices.numel() == 1:
                keep.append(sorted_indices.item())
                break
                
            current = sorted_indices[0]
            keep.append(current.item())
            
            if sorted_indices.numel() == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[current:current+1]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self.calculate_iou_pytorch(current_box, remaining_boxes)
            
            # Keep boxes with IoU less than threshold
            valid_indices = ious < self.iou_threshold
            sorted_indices = sorted_indices[1:][valid_indices]
        
        return torch.tensor(keep, dtype=torch.int64)
    
    def calculate_iou_pytorch(self, box1, boxes2):
        """
        Calculate IoU between one box and multiple boxes
        """
        # Expand box1 to match boxes2 shape
        box1_expanded = box1.expand_as(boxes2)
        
        # Calculate intersection
        x1 = torch.max(box1_expanded[:, 0], boxes2[:, 0])
        y1 = torch.max(box1_expanded[:, 1], boxes2[:, 1])
        x2 = torch.min(box1_expanded[:, 2], boxes2[:, 2])
        y2 = torch.min(box1_expanded[:, 3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (box1_expanded[:, 2] - box1_expanded[:, 0]) * (box1_expanded[:, 3] - box1_expanded[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate IoU
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-8)
        
        return iou


def convert_to_onnx():
    """
    Convert EfficientDet to ONNX with proper post-processing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weights', type=str, default=None, help='path to weights file')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--output_path', type=str, default='efficientdet_standalone.onnx', help='output onnx path')
    
    args = parser.parse_args()
    
    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    confidence_threshold = args.confidence_threshold
    
    # Input size based on compound coefficient
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]
    
    # Create model
    model = EfficientDetONNXModel(
        compound_coef=compound_coef,
        num_classes=80,  # Change this to your number of classes
        ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    )
    
    # Load weights if provided
    if args.weights:
        try:
            # Load the state dict
            state_dict = torch.load(args.weights, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load only the backbone weights
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                else:
                    print(f"Skipping {k}: shape mismatch or not found")
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded weights from {args.weights}")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with random weights...")
    
    model.eval()
    
    # Set thresholds
    model.threshold = confidence_threshold
    model.iou_threshold = nms_threshold
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Converting EfficientDet D{compound_coef} to ONNX...")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size', 1: 'num_detections'},
            'scores': {0: 'batch_size', 1: 'num_detections'},
            'labels': {0: 'batch_size', 1: 'num_detections'}
        }
    )
    
    print(f"ONNX model saved to: {args.output_path}")
    print("Model outputs:")
    print("  - boxes: [batch_size, num_detections, 4] - (x1, y1, x2, y2)")
    print("  - scores: [batch_size, num_detections] - confidence scores")
    print("  - labels: [batch_size, num_detections] - class labels")


if __name__ == '__main__':
    convert_to_onnx()