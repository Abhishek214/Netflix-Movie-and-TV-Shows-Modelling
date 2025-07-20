"""
Simplified COCO Evaluation Script for EfficientDet ONNX Model

This is a streamlined version that's easier to debug and modify.

Usage:
    python simple_coco_eval_onnx.py --onnx_path model.onnx --coco_path /path/to/coco
"""

import argparse
import json
import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def load_coco_dataset(coco_path, max_images=None):
    """Load COCO validation dataset"""
    annotation_file = os.path.join(coco_path, 'annotations/instances_val2017.json')
    image_dir = os.path.join(coco_path, 'val2017')
    
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    
    if max_images:
        image_ids = image_ids[:max_images]
    
    return coco, image_ids, image_dir


def preprocess_image(image_path, target_size=768):
    """Preprocess image for EfficientDet inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    image_resized = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to numpy and normalize
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # CHW format + batch dimension
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array, original_size


def postprocess_onnx_outputs(outputs, original_size, target_size, confidence_threshold=0.05):
    """Convert ONNX outputs to COCO detection format"""
    detections = []
    
    if len(outputs) == 0:
        return detections
    
    # Get the main detection output (format may vary)
    raw_detections = outputs[0]
    
    if raw_detections.size == 0:
        return detections
    
    original_w, original_h = original_size
    scale_x = original_w / target_size
    scale_y = original_h / target_size
    
    # Process each detection
    for detection in raw_detections:
        if len(detection) >= 7:
            # Typical format: [batch_id, x1, y1, x2, y2, confidence, class_id]
            try:
                batch_id, x1, y1, x2, y2, confidence, class_id = detection[:7]
                
                if confidence >= confidence_threshold:
                    # Scale coordinates back to original image
                    x1_orig = max(0, min(x1 * scale_x, original_w))
                    y1_orig = max(0, min(y1 * scale_y, original_h))
                    x2_orig = max(0, min(x2 * scale_x, original_w))
                    y2_orig = max(0, min(y2 * scale_y, original_h))
                    
                    width = x2_orig - x1_orig
                    height = y2_orig - y1_orig
                    
                    if width > 0 and height > 0:
                        detections.append({
                            'bbox': [x1_orig, y1_orig, width, height],  # COCO format
                            'score': float(confidence),
                            'category_id': int(class_id)
                        })
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse detection: {e}")
                continue
    
    return detections


def run_onnx_inference(onnx_path, image_array):
    """Run inference with ONNX model"""
    # Create session if not already created (for efficiency, this should be done once)
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: image_array})
    
    return outputs


def create_coco_class_mapping():
    """Create mapping between model output classes and COCO category IDs"""
    # COCO category IDs (not consecutive!)
    coco_category_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    
    # Create mapping from model class index to COCO category ID
    class_to_coco = {}
    for i, coco_id in enumerate(coco_category_ids):
        class_to_coco[i] = coco_id
    
    return class_to_coco


def evaluate_model(onnx_path, coco_path, max_images=None, confidence_threshold=0.05, target_size=768):
    """Main evaluation function"""
    print(f"Loading COCO dataset from: {coco_path}")
    coco_gt, image_ids, image_dir = load_coco_dataset(coco_path, max_images)
    
    print(f"Loading ONNX model from: {onnx_path}")
    
    # Create ONNX session once
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {session.get_inputs()[0].shape}")
    print(f"Target image size: {target_size}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Get class mapping
    class_to_coco = create_coco_class_mapping()
    
    # Run inference on all images
    results = []
    inference_times = []
    
    print(f"Processing {len(image_ids)} images...")
    
    for idx, image_id in enumerate(tqdm(image_ids)):
        # Get image info
        img_info = coco_gt.loadImgs([image_id])[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Preprocess image
            image_array, original_size = preprocess_image(image_path, target_size)
            
            # Run inference
            start_time = time.time()
            outputs = session.run(None, {input_name: image_array})
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Post-process outputs
            detections = postprocess_onnx_outputs(
                outputs, original_size, target_size, confidence_threshold
            )
            
            # Convert to COCO format
            for det in detections:
                model_class_id = det['category_id']
                if model_class_id in class_to_coco:
                    coco_category_id = class_to_coco[model_class_id]
                    
                    result = {
                        'image_id': image_id,
                        'category_id': coco_category_id,
                        'bbox': det['bbox'],
                        'score': det['score']
                    }
                    results.append(result)
        
        except Exception as e:
            print(f"Error processing image {img_info['file_name']}: {e}")
            continue
        
        # Print progress every 100 images
        if (idx + 1) % 100 == 0:
            avg_time = np.mean(inference_times[-100:]) if len(inference_times) >= 100 else np.mean(inference_times)
            print(f"Processed {idx + 1}/{len(image_ids)} images. Avg time: {avg_time:.4f}s")
    
    # Save results
    results_file = 'coco_results_onnx.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print(f"\nInference completed!")
    print(f"Total images processed: {len(inference_times)}")
    print(f"Total detections: {len(results)}")
    print(f"Average inference time: {np.mean(inference_times):.4f}s")
    print(f"FPS: {1.0/np.mean(inference_times):.2f}")
    print(f"Results saved to: {results_file}")
    
    if len(results) == 0:
        print("No valid detections found! Check your model and confidence threshold.")
        return None
    
    # Evaluate with COCO API
    print("\nRunning COCO evaluation...")
    try:
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP_50_95': coco_eval.stats[0],
            'mAP_50': coco_eval.stats[1], 
            'mAP_75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'avg_inference_time': np.mean(inference_times),
            'fps': 1.0/np.mean(inference_times)
        }
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"mAP @ IoU=0.50:0.95: {metrics['mAP_50_95']:.4f}")
        print(f"mAP @ IoU=0.50:     {metrics['mAP_50']:.4f}")
        print(f"mAP @ IoU=0.75:     {metrics['mAP_75']:.4f}")
        print(f"mAP (small):        {metrics['mAP_small']:.4f}")
        print(f"mAP (medium):       {metrics['mAP_medium']:.4f}")
        print(f"mAP (large):        {metrics['mAP_large']:.4f}")
        print(f"Average FPS:        {metrics['fps']:.2f}")
        print("="*60)
        
        return metrics
        
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Simple COCO Evaluation for EfficientDet ONNX')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--coco_path', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--max_images', type=int, default=None, help='Max images to evaluate (for testing)')
    parser.add_argument('--confidence_threshold', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--target_size', type=int, default=768, help='Input image size (768 for D2)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")
    
    if not os.path.exists(args.coco_path):
        raise FileNotFoundError(f"COCO dataset path not found: {args.coco_path}")
    
    # Run evaluation
    metrics = evaluate_model(
        onnx_path=args.onnx_path,
        coco_path=args.coco_path,
        max_images=args.max_images,
        confidence_threshold=args.confidence_threshold,
        target_size=args.target_size
    )
    
    if metrics:
        print("Evaluation completed successfully!")
    else:
        print("Evaluation failed!")


if __name__ == '__main__':
    main()
