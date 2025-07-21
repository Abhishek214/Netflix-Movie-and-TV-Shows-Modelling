"""
Updated evaluation script for ONNX model with full post-processing
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


def preprocess_image(image_path, target_size=768):
    """Preprocess image for EfficientDet inference"""
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


def convert_detections_to_coco(detections, image_id, original_size, target_size):
    """Convert ONNX detections to COCO format"""
    if len(detections) == 0:
        return []
    
    original_w, original_h = original_size
    scale_x = original_w / target_size
    scale_y = original_h / target_size
    
    results = []
    
    # COCO category mapping (model class index -> COCO category ID)
    coco_category_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]
    
    for detection in detections:
        # Format: [batch_id, x1, y1, x2, y2, confidence, class_id]
        if len(detection) >= 7:
            batch_id, x1, y1, x2, y2, confidence, class_id = detection[:7]
            
            # Scale coordinates back to original image
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            
            width = x2_orig - x1_orig
            height = y2_orig - y1_orig
            
            # Convert class index to COCO category ID
            class_idx = int(class_id)
            if 0 <= class_idx < len(coco_category_ids):
                coco_category_id = coco_category_ids[class_idx]
                
                results.append({
                    'image_id': int(image_id),
                    'category_id': coco_category_id,
                    'bbox': [float(x1_orig), float(y1_orig), float(width), float(height)],
                    'score': float(confidence)
                })
    
    return results


def evaluate_onnx_model(onnx_path, coco_path, target_size=768, max_images=None):
    """Evaluate ONNX model with full post-processing"""
    
    # Load COCO dataset
    annotation_file = os.path.join(coco_path, 'annotations/instances_val2017.json')
    image_dir = os.path.join(coco_path, 'val2017')
    
    coco_gt = COCO(annotation_file)
    image_ids = coco_gt.getImgIds()
    
    if max_images:
        image_ids = image_ids[:max_images]
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    print(f"Model: {onnx_path}")
    print(f"Input: {input_name} - {session.get_inputs()[0].shape}")
    print(f"Output: {session.get_outputs()[0].name} - {session.get_outputs()[0].shape}")
    print(f"Processing {len(image_ids)} images...")
    
    # Run inference
    results = []
    inference_times = []
    
    for image_id in tqdm(image_ids):
        img_info = coco_gt.loadImgs([image_id])[0]
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        try:
            # Preprocess
            image_array, original_size = preprocess_image(image_path, target_size)
            
            # Inference
            start_time = time.time()
            detections = session.run(None, {input_name: image_array})[0]
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert to COCO format
            coco_detections = convert_detections_to_coco(
                detections, image_id, original_size, target_size
            )
            results.extend(coco_detections)
            
        except Exception as e:
            print(f"Error processing {img_info['file_name']}: {e}")
            continue
    
    # Save and evaluate
    results_file = 'coco_results_onnx_postproc.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    print(f"\nResults:")
    print(f"Images processed: {len(inference_times)}")
    print(f"Total detections: {len(results)}")
    print(f"Average inference time: {np.mean(inference_times):.4f}s")
    print(f"FPS: {1.0/np.mean(inference_times):.2f}")
    
    if len(results) > 0:
        # COCO evaluation
        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        print(f"\nmAP @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"mAP @ IoU=0.50:     {coco_eval.stats[1]:.4f}")
        print(f"mAP @ IoU=0.75:     {coco_eval.stats[2]:.4f}")
    else:
        print("No detections found!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model with post-processing')
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--target_size', type=int, default=768)
    parser.add_argument('--max_images', type=int, default=None)
    
    args = parser.parse_args()
    
    evaluate_onnx_model(args.onnx_path, args.coco_path, args.target_size, args.max_images)


if __name__ == '__main__':
    main()
