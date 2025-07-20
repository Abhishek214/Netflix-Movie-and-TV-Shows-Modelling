#!/usr/bin/env python3
"""
COCO Evaluation Script for ONNX Models
Evaluates ONNX EfficientDet models on COCO dataset and calculates mAP metrics
"""

import os
import json
import argparse
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import onnxruntime as ort

# COCO evaluation imports
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Error: pycocotools not found. Install with: pip install pycocotools")
    exit(1)

class ONNXCocoEvaluator:
    def __init__(self, model_path, coco_path, input_size=768, score_threshold=0.05, nms_threshold=0.5):
        """
        Initialize ONNX COCO evaluator
        
        Args:
            model_path: Path to ONNX model
            coco_path: Path to COCO annotations file
            input_size: Model input size (768 for EfficientDet D2)
            score_threshold: Minimum confidence score
            nms_threshold: NMS threshold
        """
        self.model_path = model_path
        self.coco_path = coco_path
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        # COCO class names
        self.coco_class_names = [
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
        
        # Initialize ONNX model
        self._load_model()
        
        # Initialize COCO API
        self._load_coco_data()
    
    def _load_model(self):
        """Load ONNX model"""
        print(f"Loading ONNX model: {self.model_path}")
        
        try:
            # Set up providers
            providers = []
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            print(f"✓ Model loaded successfully")
            print(f"✓ Using providers: {self.session.get_providers()}")
            print(f"✓ Input: {self.input_name}")
            print(f"✓ Outputs: {self.output_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_coco_data(self):
        """Load COCO dataset"""
        print(f"Loading COCO data: {self.coco_path}")
        
        try:
            self.coco_gt = COCO(self.coco_path)
            self.image_ids = list(self.coco_gt.imgs.keys())
            
            print(f"✓ COCO data loaded successfully")
            print(f"✓ Number of images: {len(self.image_ids)}")
            print(f"✓ Number of categories: {len(self.coco_gt.cats)}")
            
            # Get category mapping
            self.coco_categories = self.coco_gt.loadCats(self.coco_gt.getCatIds())
            self.category_mapping = {cat['id']: cat['name'] for cat in self.coco_categories}
            
        except Exception as e:
            print(f"❌ Error loading COCO data: {e}")
            raise
    
    def preprocess_image(self, image_path, target_size=None):
        """
        Preprocess image for ONNX model
        
        Args:
            image_path: Path to image
            target_size: Target size (uses self.input_size if None)
            
        Returns:
            preprocessed_array: Preprocessed image array
            original_image: Original PIL image
            scale_factors: Scale factors for coordinate conversion
        """
        if target_size is None:
            target_size = self.input_size
            
        try:
            # Load image
            original_image = Image.open(image_path).convert('RGB')
            original_width, original_height = original_image.size
            
            # Resize image
            resized_image = original_image.resize((target_size, target_size), Image.BILINEAR)
            
            # Convert to numpy array (float32)
            image_array = np.array(resized_image, dtype=np.float32)
            
            # Normalize to [0, 1]
            image_array = image_array / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            image_array = (image_array - mean) / std
            
            # Convert to CHW format and add batch dimension
            image_array = np.transpose(image_array, (2, 0, 1))
            image_array = np.expand_dims(image_array, axis=0)
            
            # Calculate scale factors
            scale_x = original_width / target_size
            scale_y = original_height / target_size
            
            return image_array, original_image, (scale_x, scale_y)
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None, None, None
    
    def run_inference(self, image_array):
        """Run inference on preprocessed image array"""
        try:
            outputs = self.session.run(None, {self.input_name: image_array})
            return outputs
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def postprocess_detections(self, outputs, scale_factors, original_size):
        """
        Post-process model outputs to COCO format
        
        Args:
            outputs: Model outputs [boxes, scores, labels]
            scale_factors: Scale factors from preprocessing
            original_size: Original image size (width, height)
            
        Returns:
            detections: List of detection dictionaries in COCO format
        """
        detections = []
        
        if outputs is None or len(outputs) < 3:
            return detections
        
        boxes = outputs[0]  # [batch_size, num_detections, 4]
        scores = outputs[1]  # [batch_size, num_detections]
        labels = outputs[2]  # [batch_size, num_detections]
        
        scale_x, scale_y = scale_factors
        original_width, original_height = original_size
        
        # Remove batch dimension
        if len(boxes.shape) > 2:
            boxes = boxes[0]
        if len(scores.shape) > 1:
            scores = scores[0]
        if len(labels.shape) > 1:
            labels = labels[0]
        
        # Filter by score threshold
        valid_mask = scores >= self.score_threshold
        
        if not np.any(valid_mask):
            return detections
        
        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        
        # Convert to COCO format
        for box, score, label in zip(valid_boxes, valid_scores, valid_labels):
            x1, y1, x2, y2 = box
            
            # Scale coordinates back to original image
            x1_scaled = float(x1 * scale_x)
            y1_scaled = float(y1 * scale_y)
            x2_scaled = float(x2 * scale_x)
            y2_scaled = float(y2 * scale_y)
            
            # Clamp coordinates to image bounds
            x1_scaled = max(0, min(x1_scaled, original_width))
            y1_scaled = max(0, min(y1_scaled, original_height))
            x2_scaled = max(0, min(x2_scaled, original_width))
            y2_scaled = max(0, min(y2_scaled, original_height))
            
            # Calculate width and height
            width = x2_scaled - x1_scaled
            height = y2_scaled - y1_scaled
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            # Convert label to COCO category ID
            label_idx = int(label)
            if label_idx < len(self.coco_categories):
                category_id = self.coco_categories[label_idx]['id']
            else:
                continue  # Skip invalid categories
            
            detections.append({
                'bbox': [x1_scaled, y1_scaled, width, height],  # COCO format: [x, y, w, h]
                'score': float(score),
                'category_id': category_id
            })
        
        return detections
    
    def evaluate_on_coco(self, image_dir, max_images=None, save_results=True):
        """
        Evaluate model on COCO dataset
        
        Args:
            image_dir: Directory containing COCO images
            max_images: Maximum number of images to evaluate (None for all)
            save_results: Whether to save detection results
            
        Returns:
            evaluation_results: Dictionary containing mAP metrics
        """
        print(f"Starting COCO evaluation...")
        print(f"Image directory: {image_dir}")
        print(f"Score threshold: {self.score_threshold}")
        print(f"Input size: {self.input_size}")
        
        # Prepare results
        coco_results = []
        total_time = 0
        processed_images = 0
        
        # Get subset of images if specified
        eval_image_ids = self.image_ids[:max_images] if max_images else self.image_ids
        
        print(f"Evaluating on {len(eval_image_ids)} images...")
        
        # Process images
        for img_id in tqdm(eval_image_ids, desc="Processing images"):
            try:
                # Get image info
                img_info = self.coco_gt.loadImgs([img_id])[0]
                image_path = os.path.join(image_dir, img_info['file_name'])
                
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Preprocess image
                image_array, original_image, scale_factors = self.preprocess_image(image_path)
                
                if image_array is None:
                    continue
                
                # Run inference
                start_time = time.time()
                outputs = self.run_inference(image_array)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                if outputs is None:
                    continue
                
                # Post-process detections
                detections = self.postprocess_detections(
                    outputs, 
                    scale_factors, 
                    original_image.size
                )
                
                # Add image_id to each detection and append to results
                for detection in detections:
                    detection['image_id'] = img_id
                    coco_results.append(detection)
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")
                continue
        
        print(f"✓ Processed {processed_images} images")
        print(f"✓ Total detections: {len(coco_results)}")
        print(f"✓ Average inference time: {total_time/processed_images:.4f}s per image")
        print(f"✓ FPS: {processed_images/total_time:.2f}")
        
        # Save results if requested
        if save_results and coco_results:
            results_file = f"onnx_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(coco_results, f)
            print(f"✓ Results saved to: {results_file}")
        
        # Evaluate with COCO API
        if coco_results:
            evaluation_results = self._run_coco_evaluation(coco_results)
        else:
            print("❌ No valid results to evaluate")
            evaluation_results = {}
        
        return evaluation_results
    
    def _run_coco_evaluation(self, coco_results):
        """Run COCO evaluation using pycocotools"""
        print("\nRunning COCO evaluation...")
        
        try:
            # Create results in COCO format
            coco_dt = self.coco_gt.loadRes(coco_results)
            
            # Initialize COCO evaluator
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'mAP': coco_eval.stats[0],          # mAP @ IoU=0.50:0.95
                'mAP_50': coco_eval.stats[1],       # mAP @ IoU=0.50
                'mAP_75': coco_eval.stats[2],       # mAP @ IoU=0.75
                'mAP_small': coco_eval.stats[3],    # mAP for small objects
                'mAP_medium': coco_eval.stats[4],   # mAP for medium objects
                'mAP_large': coco_eval.stats[5],    # mAP for large objects
                'AR_1': coco_eval.stats[6],         # AR given 1 detection per image
                'AR_10': coco_eval.stats[7],        # AR given 10 detections per image
                'AR_100': coco_eval.stats[8],       # AR given 100 detections per image
                'AR_small': coco_eval.stats[9],     # AR for small objects
                'AR_medium': coco_eval.stats[10],   # AR for medium objects
                'AR_large': coco_eval.stats[11],    # AR for large objects
            }
            
            # Print summary
            print("\n" + "="*60)
            print("COCO EVALUATION RESULTS")
            print("="*60)
            print(f"mAP (IoU=0.50:0.95): {metrics['mAP']:.4f}")
            print(f"mAP (IoU=0.50):      {metrics['mAP_50']:.4f}")
            print(f"mAP (IoU=0.75):      {metrics['mAP_75']:.4f}")
            print(f"mAP (small):         {metrics['mAP_small']:.4f}")
            print(f"mAP (medium):        {metrics['mAP_medium']:.4f}")
            print(f"mAP (large):         {metrics['mAP_large']:.4f}")
            print("-"*60)
            print(f"AR (max=1):          {metrics['AR_1']:.4f}")
            print(f"AR (max=10):         {metrics['AR_10']:.4f}")
            print(f"AR (max=100):        {metrics['AR_100']:.4f}")
            print(f"AR (small):          {metrics['AR_small']:.4f}")
            print(f"AR (medium):         {metrics['AR_medium']:.4f}")
            print(f"AR (large):          {metrics['AR_large']:.4f}")
            print("="*60)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error during COCO evaluation: {e}")
            return {}
    
    def evaluate_single_image(self, image_path, visualize=False, save_path=None):
        """
        Evaluate model on a single image (for debugging)
        
        Args:
            image_path: Path to image
            visualize: Whether to create visualization
            save_path: Path to save visualization
            
        Returns:
            detections: List of detections
        """
        print(f"Evaluating single image: {image_path}")
        
        # Preprocess
        image_array, original_image, scale_factors = self.preprocess_image(image_path)
        
        if image_array is None:
            return []
        
        # Run inference
        start_time = time.time()
        outputs = self.run_inference(image_array)
        end_time = time.time()
        
        print(f"Inference time: {end_time - start_time:.4f}s")
        
        if outputs is None:
            return []
        
        # Post-process
        detections = self.postprocess_detections(
            outputs, 
            scale_factors, 
            original_image.size
        )
        
        print(f"Found {len(detections)} detections")
        
        # Print detections
        for i, det in enumerate(detections):
            bbox = det['bbox']
            score = det['score']
            cat_id = det['category_id']
            cat_name = self.category_mapping.get(cat_id, f"category_{cat_id}")
            print(f"  {i+1}. {cat_name}: {score:.3f} at [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Visualize if requested
        if visualize and detections:
            self._visualize_detections(original_image, detections, save_path)
        
        return detections
    
    def _visualize_detections(self, image, detections, save_path=None):
        """Visualize detections on image"""
        try:
            from PIL import ImageDraw, ImageFont
            
            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Colors for different categories
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
            
            for detection in detections:
                x, y, w, h = detection['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
                score = detection['score']
                cat_id = detection['category_id']
                cat_name = self.category_mapping.get(cat_id, f"cat_{cat_id}")
                
                # Select color
                color = colors[cat_id % len(colors)]
                
                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{cat_name}: {score:.2f}"
                draw.text((x1 + 2, y1 - 25), label, fill=color, font=font)
            
            if save_path:
                result_image.save(save_path)
                print(f"✓ Visualization saved to: {save_path}")
            else:
                result_image.show()
                
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    parser = argparse.ArgumentParser(description='COCO Evaluation for ONNX Models')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--coco_path', type=str, required=True, help='Path to COCO annotations file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing COCO images')
    parser.add_argument('--input_size', type=int, default=768, help='Model input size')
    parser.add_argument('--score_threshold', type=float, default=0.05, help='Score threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS threshold')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum images to evaluate')
    parser.add_argument('--single_image', type=str, default=None, help='Evaluate single image (for debugging)')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save detection results')
    
    args = parser.parse_args()
    
    print("ONNX COCO Evaluator")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"COCO annotations: {args.coco_path}")
    print(f"Image directory: {args.image_dir}")
    print(f"Input size: {args.input_size}")
    print(f"Score threshold: {args.score_threshold}")
    
    # Initialize evaluator
    try:
        evaluator = ONNXCocoEvaluator(
            model_path=args.model,
            coco_path=args.coco_path,
            input_size=args.input_size,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold
        )
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return 1
    
    # Single image evaluation (for debugging)
    if args.single_image:
        detections = evaluator.evaluate_single_image(
            args.single_image, 
            visualize=args.visualize,
            save_path="single_image_result.jpg" if args.visualize else None
        )
        return 0
    
    # Full COCO evaluation
    try:
        results = evaluator.evaluate_on_coco(
            image_dir=args.image_dir,
            max_images=args.max_images,
            save_results=args.save_results
        )
        
        if results:
            print(f"\n🎉 Evaluation completed successfully!")
            print(f"📊 Final mAP: {results.get('mAP', 0):.4f}")
        else:
            print("❌ Evaluation failed")
            return 1
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())