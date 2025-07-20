#!/usr/bin/env python3
"""
Inference script for RAW ONNX EfficientDet models
Implements proper post-processing to fix the 0 mAP issue
"""

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import math
import argparse

class RawONNXInference:
    def __init__(self, onnx_model_path, input_size=768, num_classes=80):
        """
        Initialize inference for RAW ONNX model (without post-processing)
        """
        self.onnx_model_path = onnx_model_path
        self.input_size = input_size
        self.num_classes = num_classes
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Load ONNX model
        self._load_model()
        
        # Generate anchors
        self.anchors = self._generate_anchors()
        
    def _load_model(self):
        """Load ONNX model"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_model_path, providers=providers)
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            print(f"✓ RAW ONNX model loaded")
            print(f"✓ Input: {self.input_name}")
            print(f"✓ Outputs: {self.output_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _generate_anchors(self):
        """
        Generate anchor boxes for EfficientDet
        """
        print("Generating anchors...")
        
        # EfficientDet anchor configuration
        anchor_scale = 4.0
        aspect_ratios = [0.5, 1.0, 2.0]
        num_scales = 3
        pyramid_levels = [3, 4, 5, 6, 7]  # P3 to P7
        
        anchors = []
        
        for level in pyramid_levels:
            stride = 2 ** level
            feature_size = self.input_size // stride
            
            # Create grid
            shifts_x = np.arange(0, feature_size) * stride
            shifts_y = np.arange(0, feature_size) * stride
            shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            
            # Generate anchors for each aspect ratio and scale
            for aspect_ratio in aspect_ratios:
                for scale_idx in range(num_scales):
                    scale = anchor_scale * (2 ** (scale_idx / num_scales))
                    
                    # Calculate anchor size
                    area = (stride * scale) ** 2
                    w = np.sqrt(area / aspect_ratio)
                    h = w * aspect_ratio
                    
                    # Create anchors
                    anchors_level = np.stack([
                        shift_x - w/2,  # x1
                        shift_y - h/2,  # y1
                        shift_x + w/2,  # x2
                        shift_y + h/2   # y2
                    ], axis=1)
                    
                    anchors.append(anchors_level)
        
        all_anchors = np.concatenate(anchors, axis=0)
        print(f"✓ Generated {len(all_anchors)} anchors")
        
        return all_anchors
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize
        resized_image = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Convert to array
        image_array = np.array(resized_image, dtype=np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        # Convert to CHW and add batch dimension
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array, image, original_size
    
    def run_inference(self, image_array):
        """Run inference on preprocessed image"""
        try:
            outputs = self.session.run(None, {self.input_name: image_array})
            return outputs
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def decode_boxes(self, box_regression, anchors):
        """
        Decode box regression predictions using anchors
        """
        # Apply regression to anchors
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        dx = box_regression[:, 0]
        dy = box_regression[:, 1] 
        dw = box_regression[:, 2]
        dh = box_regression[:, 3]
        
        # Decode
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights
        
        # Convert to x1, y1, x2, y2
        pred_boxes = np.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h
        ], axis=1)
        
        return pred_boxes
    
    def apply_nms(self, boxes, scores, labels, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression
        """
        # Convert to torch tensors for easier NMS
        boxes_tensor = torch.from_numpy(boxes)
        scores_tensor = torch.from_numpy(scores)
        
        # Apply NMS
        keep_indices = []
        for class_id in np.unique(labels):
            class_mask = labels == class_id
            if not np.any(class_mask):
                continue
                
            class_boxes = boxes_tensor[class_mask]
            class_scores = scores_tensor[class_mask]
            
            # PyTorch NMS
            nms_indices = torch.ops.torchvision.nms(class_boxes, class_scores, iou_threshold)
            
            # Convert back to original indices
            original_indices = np.where(class_mask)[0]
            keep_indices.extend(original_indices[nms_indices.numpy()])
        
        return np.array(keep_indices)
    
    def postprocess_detections(self, raw_outputs, original_size, score_threshold=0.3, nms_threshold=0.5):
        """
        Post-process raw model outputs
        """
        if raw_outputs is None or len(raw_outputs) < 2:
            return []
        
        # Extract outputs
        box_regression = raw_outputs[0]  # [batch, num_anchors, 4]
        classification = raw_outputs[1]   # [batch, num_anchors, num_classes]
        
        # Remove batch dimension
        if len(box_regression.shape) > 2:
            box_regression = box_regression[0]
        if len(classification.shape) > 2:
            classification = classification[0]
        
        print(f"Raw regression shape: {box_regression.shape}")
        print(f"Raw classification shape: {classification.shape}")
        print(f"Anchors shape: {self.anchors.shape}")
        
        # Ensure we have matching number of anchors
        num_predictions = min(len(box_regression), len(classification), len(self.anchors))
        box_regression = box_regression[:num_predictions]
        classification = classification[:num_predictions]
        anchors = self.anchors[:num_predictions]
        
        # Apply sigmoid to classification scores
        scores = 1.0 / (1.0 + np.exp(-classification))  # Sigmoid
        
        # Get class predictions
        class_scores = np.max(scores, axis=1)
        class_labels = np.argmax(scores, axis=1)
        
        # Filter by score threshold
        valid_mask = class_scores >= score_threshold
        
        if not np.any(valid_mask):
            print(f"No detections above threshold {score_threshold}")
            return []
        
        valid_regression = box_regression[valid_mask]
        valid_scores = class_scores[valid_mask]
        valid_labels = class_labels[valid_mask]
        valid_anchors = anchors[valid_mask]
        
        # Decode boxes
        decoded_boxes = self.decode_boxes(valid_regression, valid_anchors)
        
        # Clip boxes to image bounds
        decoded_boxes[:, 0] = np.clip(decoded_boxes[:, 0], 0, self.input_size)
        decoded_boxes[:, 1] = np.clip(decoded_boxes[:, 1], 0, self.input_size)
        decoded_boxes[:, 2] = np.clip(decoded_boxes[:, 2], 0, self.input_size)
        decoded_boxes[:, 3] = np.clip(decoded_boxes[:, 3], 0, self.input_size)
        
        # Apply NMS
        try:
            keep_indices = self.apply_nms(decoded_boxes, valid_scores, valid_labels, nms_threshold)
            final_boxes = decoded_boxes[keep_indices]
            final_scores = valid_scores[keep_indices]
            final_labels = valid_labels[keep_indices]
        except:
            # Fallback: simple score-based filtering
            top_k = min(100, len(valid_scores))
            top_indices = np.argsort(valid_scores)[-top_k:]
            final_boxes = decoded_boxes[top_indices]
            final_scores = valid_scores[top_indices]
            final_labels = valid_labels[top_indices]
        
        # Scale boxes to original image size
        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size
        
        detections = []
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = box
            
            # Scale to original image
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            # Ensure valid box
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"class_{int(label)}"
                
                detections.append({
                    'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    'confidence': float(score),
                    'class_id': int(label),
                    'class_name': class_name
                })
        
        return detections
    
    def inference(self, image_path, score_threshold=0.3, nms_threshold=0.5):
        """
        Complete inference pipeline
        """
        print(f"Running inference on: {image_path}")
        
        # Preprocess
        image_array, original_image, original_size = self.preprocess_image(image_path)
        
        # Run inference
        raw_outputs = self.run_inference(image_array)
        
        if raw_outputs is None:
            return [], original_image
        
        # Post-process
        detections = self.postprocess_detections(
            raw_outputs, 
            original_size, 
            score_threshold, 
            nms_threshold
        )
        
        print(f"Found {len(detections)} detections")
        return detections, original_image
    
    def visualize_detections(self, image, detections, save_path=None):
        """Visualize detections"""
        if not detections:
            print("No detections to visualize")
            return image
        
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            color = colors[detection['class_id'] % len(colors)]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            draw.text((x1 + 2, y1 - 25), label, fill=color)
        
        if save_path:
            result_image.save(save_path)
            print(f"Visualization saved to: {save_path}")
        
        return result_image

def main():
    parser = argparse.ArgumentParser(description='Raw ONNX EfficientDet Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to RAW ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--output', type=str, default='raw_inference_result.jpg', help='Output path')
    parser.add_argument('--threshold', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--input_size', type=int, default=768, help='Input size')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inferencer = RawONNXInference(
            onnx_model_path=args.model,
            input_size=args.input_size
        )
        
        # Run inference
        detections, original_image = inferencer.inference(
            args.image,
            score_threshold=args.threshold
        )
        
        print(f"\n🎉 Inference completed!")
        print(f"📊 Found {len(detections)} detections:")
        
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
        
        # Visualize
        result_image = inferencer.visualize_detections(
            original_image,
            detections,
            save_path=args.output
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()