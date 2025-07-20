#!/usr/bin/env python3
"""
Simple inference script for standalone EfficientDet ONNX model
"""

import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import argparse

class SimpleONNXInference:
    def __init__(self, model_path, input_size=768, num_classes=80):
        """
        Initialize ONNX inference
        
        Args:
            model_path: Path to ONNX model
            input_size: Input image size
            num_classes: Number of classes
        """
        self.model_path = model_path
        self.input_size = input_size
        self.num_classes = num_classes
        
        # COCO class names (change this for your custom classes)
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
        
        # Initialize ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Providers: {self.session.get_providers()}")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"✓ Input: {self.input_name}")
        print(f"✓ Outputs: {self.output_names}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize
        image_resized = image.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Convert to numpy
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to CHW format and add batch dimension
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        # Calculate scale factors for coordinate conversion
        scale_x = original_size[0] / self.input_size
        scale_y = original_size[1] / self.input_size
        
        return image_array, image, (scale_x, scale_y)
    
    def postprocess_detections(self, boxes, scores, labels, scale_factors, score_threshold=0.3):
        """Post-process model outputs"""
        detections = []
        scale_x, scale_y = scale_factors
        
        # Remove batch dimension
        if len(boxes.shape) > 2:
            boxes = boxes[0]
        if len(scores.shape) > 1:
            scores = scores[0]
        if len(labels.shape) > 1:
            labels = labels[0]
        
        # Filter by score threshold
        valid_mask = scores >= score_threshold
        
        if not np.any(valid_mask):
            return detections
        
        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Convert to detections
        for box, score, label in zip(valid_boxes, valid_scores, valid_labels):
            x1, y1, x2, y2 = box
            
            # Scale coordinates back to original image
            x1_scaled = float(x1 * scale_x)
            y1_scaled = float(y1 * scale_y)
            x2_scaled = float(x2 * scale_x)
            y2_scaled = float(y2 * scale_y)
            
            class_id = int(label)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            detections.append({
                'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                'confidence': float(score),
                'class_id': class_id,
                'class_name': class_name
            })
        
        return detections
    
    def run_inference(self, image_path, score_threshold=0.3):
        """Run inference on image"""
        try:
            # Preprocess
            input_data, original_image, scale_factors = self.preprocess_image(image_path)
            
            print(f"✓ Input shape: {input_data.shape}")
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_data})
            
            print(f"✓ Inference completed")
            print(f"✓ Output shapes: {[out.shape for out in outputs]}")
            
            # Extract outputs
            boxes = outputs[0]
            scores = outputs[1] 
            labels = outputs[2]
            
            # Post-process
            detections = self.postprocess_detections(
                boxes, scores, labels, scale_factors, score_threshold
            )
            
            return detections, original_image
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            raise
    
    def visualize_detections(self, image, detections, output_path=None):
        """Visualize detections on image"""
        if not detections:
            print("No detections to visualize")
            return image
        
        # Create copy for drawing
        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Color palette
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Select color
            color = colors[detection['class_id'] % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text background size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
        
        if output_path:
            result_image.save(output_path)
            print(f"✓ Result saved to: {output_path}")
        
        return result_image

def main():
    parser = argparse.ArgumentParser(description='EfficientDet ONNX Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='result.jpg', help='Output image path')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--input_size', type=int, default=768, help='Model input size')
    
    args = parser.parse_args()
    
    print("Starting EfficientDet ONNX Inference...")
    
    # Initialize inference
    detector = SimpleONNXInference(
        model_path=args.model,
        input_size=args.input_size
    )
    
    # Run inference
    detections, original_image = detector.run_inference(
        args.image, 
        score_threshold=args.threshold
    )
    
    print(f"\n✓ Found {len(detections)} detections:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
    
    # Visualize results
    result_image = detector.visualize_detections(
        original_image, 
        detections, 
        output_path=args.output
    )
    
    print(f"✓ Inference completed successfully!")

if __name__ == '__main__':
    main()