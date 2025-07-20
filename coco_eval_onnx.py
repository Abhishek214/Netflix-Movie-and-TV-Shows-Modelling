"""
COCO Evaluation Script for EfficientDet ONNX Model

This script evaluates an ONNX EfficientDet model on COCO validation dataset
and computes mAP metrics similar to the original coco_eval.py

Usage:
    python coco_eval_onnx.py --onnx_path path/to/model.onnx --data_path path/to/coco --compound_coef 2
"""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import yaml
from tqdm import tqdm


class COCODataset:
    """
    COCO Dataset for ONNX model evaluation
    """
    def __init__(self, root_dir, set_name='val2017', img_size=512, transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.img_size = img_size
        self.transform = transform
        
        self.coco = COCO(os.path.join(root_dir, 'annotations', f'instances_{set_name}.json'))
        self.image_ids = self.coco.getImgIds()
        
        # Load image info
        self.load_classes()

    def load_classes(self):
        """Load class mapping"""
        # COCO class mapping
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # Also create label names array
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.image_ids[idx])[0]
        img_path = os.path.join(self.root_dir, self.set_name, img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        
        # Store original size
        original_w, original_h = image.size
        
        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_id': self.image_ids[idx],
            'original_size': (original_w, original_h),
            'file_name': img_info['file_name']
        }


class EfficientDetONNXEvaluator:
    """
    EfficientDet ONNX Model Evaluator
    """
    def __init__(self, onnx_path, compound_coef=2, img_size=None, confidence_threshold=0.05, nms_threshold=0.5):
        self.onnx_path = onnx_path
        self.compound_coef = compound_coef
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Set image size based on compound coefficient if not provided
        if img_size is None:
            img_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
            self.img_size = img_sizes[compound_coef]
        else:
            self.img_size = img_size
            
        print(f"Using image size: {self.img_size}x{self.img_size}")
        
        # Initialize ONNX Runtime session
        print(f"Loading ONNX model from: {onnx_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input details
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        
        print(f"Model loaded successfully!")
        print(f"Input name: {self.input_name}")
        print(f"Input shape: {self.input_shape}")
        
        # Image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        """
        # Convert PIL to numpy
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to CHW format and add batch dimension
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def postprocess_detections(self, outputs, original_size, threshold=None):
        """
        Post-process model outputs to get detections
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        detections = []
        
        # Handle different output formats
        if len(outputs) >= 1:
            raw_detections = outputs[0]  # Assuming first output contains detections
        else:
            return detections
        
        original_w, original_h = original_size
        scale_x = original_w / self.img_size
        scale_y = original_h / self.img_size
        
        # Process detections
        for detection in raw_detections:
            if len(detection) >= 7:
                # Format: [batch_id, x1, y1, x2, y2, confidence, class_id]
                batch_id, x1, y1, x2, y2, confidence, class_id = detection[:7]
                
                if confidence >= threshold:
                    # Scale coordinates back to original image size
                    x1_scaled = x1 * scale_x
                    y1_scaled = y1 * scale_y
                    x2_scaled = x2 * scale_x
                    y2_scaled = y2 * scale_y
                    
                    # Ensure coordinates are within bounds
                    x1_scaled = max(0, min(x1_scaled, original_w))
                    y1_scaled = max(0, min(y1_scaled, original_h))
                    x2_scaled = max(0, min(x2_scaled, original_w))
                    y2_scaled = max(0, min(y2_scaled, original_h))
                    
                    detections.append({
                        'bbox': [x1_scaled, y1_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled],  # COCO format: [x, y, w, h]
                        'score': float(confidence),
                        'category_id': int(class_id),
                    })
        
        return detections
    
    def predict_single_image(self, image, original_size):
        """
        Run inference on a single image
        """
        # Preprocess
        input_array = self.preprocess_image(image)
        
        # Run inference
        ort_inputs = {self.input_name: input_array}
        outputs = self.ort_session.run(None, ort_inputs)
        
        # Post-process
        detections = self.postprocess_detections(outputs, original_size)
        
        return detections


def evaluate_coco(onnx_path, data_path, compound_coef=2, img_size=None, 
                  confidence_threshold=0.05, nms_threshold=0.5, max_detections=100):
    """
    Evaluate ONNX model on COCO dataset
    """
    print("="*50)
    print("COCO Evaluation for EfficientDet ONNX Model")
    print("="*50)
    
    # Initialize evaluator
    evaluator = EfficientDetONNXEvaluator(
        onnx_path=onnx_path,
        compound_coef=compound_coef,
        img_size=img_size,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold
    )
    
    # Initialize dataset
    dataset = COCODataset(
        root_dir=data_path,
        set_name='val2017',
        img_size=evaluator.img_size
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x[0]  # Since batch_size=1
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Model input size: {evaluator.img_size}x{evaluator.img_size}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print("Starting evaluation...")
    
    # Run inference
    results = []
    inference_times = []
    
    for idx, sample in enumerate(tqdm(dataloader, desc="Running inference")):
        image = sample['image']
        image_id = sample['image_id']
        original_size = sample['original_size']
        
        # Measure inference time
        start_time = time.time()
        detections = evaluator.predict_single_image(image, original_size)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Convert detections to COCO format
        for detection in detections:
            # Map class_id to COCO category_id
            class_id = detection['category_id']
            if class_id in dataset.coco_labels:
                coco_category_id = dataset.coco_labels[class_id]
            else:
                continue  # Skip if class not in COCO
                
            result = {
                'image_id': image_id,
                'category_id': coco_category_id,
                'bbox': detection['bbox'],
                'score': detection['score']
            }
            results.append(result)
        
        # Optional: limit number of images for testing
        # if idx >= 100:  # Uncomment to test on first 100 images
        #     break
    
    print(f"\nInference completed!")
    print(f"Total images processed: {len(inference_times)}")
    print(f"Average inference time: {np.mean(inference_times):.4f}s")
    print(f"FPS: {1.0/np.mean(inference_times):.2f}")
    print(f"Total detections: {len(results)}")
    
    if len(results) == 0:
        print("No detections found! Check your model and thresholds.")
        return
    
    # Save results to file
    results_file = f'results_onnx_d{compound_coef}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Evaluate using COCO API
    print("\nEvaluating with COCO API...")
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results_file)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract key metrics
    mAP = coco_eval.stats[0]  # mAP @ IoU=0.50:0.95
    mAP_50 = coco_eval.stats[1]  # mAP @ IoU=0.50
    mAP_75 = coco_eval.stats[2]  # mAP @ IoU=0.75
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP @ IoU=0.50:0.95: {mAP:.4f}")
    print(f"mAP @ IoU=0.50:     {mAP_50:.4f}")
    print(f"mAP @ IoU=0.75:     {mAP_75:.4f}")
    print(f"Average inference time: {np.mean(inference_times):.4f}s")
    print(f"FPS: {1.0/np.mean(inference_times):.2f}")
    print("="*50)
    
    return {
        'mAP': mAP,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
        'avg_inference_time': np.mean(inference_times),
        'fps': 1.0/np.mean(inference_times)
    }


def main():
    parser = argparse.ArgumentParser(description='COCO Evaluation for EfficientDet ONNX Model')
    
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to the ONNX model file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to COCO dataset directory')
    parser.add_argument('--compound_coef', type=int, default=2,
                        help='EfficientDet compound coefficient (0-7)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size (if None, determined by compound_coef)')
    parser.add_argument('--confidence_threshold', type=float, default=0.05,
                        help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='NMS threshold')
    parser.add_argument('--max_detections', type=int, default=100,
                        help='Maximum number of detections per image')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"COCO dataset directory not found: {args.data_path}")
    
    # Check for required COCO structure
    required_paths = [
        os.path.join(args.data_path, 'val2017'),
        os.path.join(args.data_path, 'annotations', 'instances_val2017.json')
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required COCO path not found: {path}")
    
    print(f"ONNX Model: {args.onnx_path}")
    print(f"COCO Dataset: {args.data_path}")
    print(f"Compound Coefficient: {args.compound_coef}")
    
    # Run evaluation
    results = evaluate_coco(
        onnx_path=args.onnx_path,
        data_path=args.data_path,
        compound_coef=args.compound_coef,
        img_size=args.img_size,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections
    )


if __name__ == '__main__':
    main()
