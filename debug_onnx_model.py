#!/usr/bin/env python3
"""
Comprehensive debugging script for ONNX EfficientDet models
Helps identify why mAP is 0.0000
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch
import json
import argparse

class ONNXModelDebugger:
    def __init__(self, onnx_model_path, pytorch_model_path=None):
        """
        Initialize debugger for ONNX model
        
        Args:
            onnx_model_path: Path to ONNX model
            pytorch_model_path: Path to original PyTorch model (optional)
        """
        self.onnx_model_path = onnx_model_path
        self.pytorch_model_path = pytorch_model_path
        
        # Load ONNX model
        self.onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]
        
        print(f"✓ ONNX model loaded: {onnx_model_path}")
        print(f"✓ Input: {self.input_name}")
        print(f"✓ Outputs: {self.output_names}")
        
        # Load PyTorch model if provided
        self.pytorch_model = None
        if pytorch_model_path and os.path.exists(pytorch_model_path):
            try:
                self.pytorch_model = self._load_pytorch_model(pytorch_model_path)
                print(f"✓ PyTorch model loaded for comparison")
            except Exception as e:
                print(f"⚠️  Could not load PyTorch model: {e}")
    
    def _load_pytorch_model(self, model_path):
        """Load original PyTorch model"""
        # This is a simplified loader - you may need to adjust based on your model structure
        checkpoint = torch.load(model_path, map_location='cpu')
        # You'll need to adapt this to your specific model architecture
        return checkpoint
    
    def preprocess_image(self, image_path, input_size=768):
        """Preprocess image for both models"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize
        resized_image = image.resize((input_size, input_size), Image.BILINEAR)
        
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
    
    def analyze_onnx_outputs(self, image_path, verbose=True):
        """Analyze raw ONNX model outputs"""
        print(f"\n{'='*60}")
        print(f"ANALYZING ONNX MODEL OUTPUTS")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        
        # Preprocess
        image_array, original_image, original_size = self.preprocess_image(image_path)
        
        # Run ONNX inference
        outputs = self.onnx_session.run(None, {self.input_name: image_array})
        
        print(f"\nRaw ONNX outputs:")
        for i, (output, name) in enumerate(zip(outputs, self.output_names)):
            print(f"  Output {i} ({name}):")
            print(f"    Shape: {output.shape}")
            print(f"    Dtype: {output.dtype}")
            print(f"    Range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"    Mean: {output.mean():.6f}")
            print(f"    Std: {output.std():.6f}")
            
            if verbose and output.size < 1000:  # Show values for small outputs
                print(f"    Sample values: {output.flatten()[:10]}")
        
        # Analyze specific outputs based on typical EfficientDet structure
        self._analyze_detection_outputs(outputs, original_size)
        
        return outputs
    
    def _analyze_detection_outputs(self, outputs, original_size):
        """Analyze detection-specific outputs"""
        print(f"\n{'-'*40}")
        print("DETECTION ANALYSIS")
        print(f"{'-'*40}")
        
        if len(outputs) >= 3:
            boxes = outputs[0]  # Assuming boxes are first output
            scores = outputs[1]  # Assuming scores are second output  
            labels = outputs[2]  # Assuming labels are third output
            
            # Remove batch dimension if present
            if len(boxes.shape) > 2:
                boxes = boxes[0]
            if len(scores.shape) > 1:
                scores = scores[0]
            if len(labels.shape) > 1:
                labels = labels[0]
            
            print(f"Boxes shape: {boxes.shape}")
            print(f"Scores shape: {scores.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Analyze score distribution
            print(f"\nScore statistics:")
            print(f"  Min score: {scores.min():.6f}")
            print(f"  Max score: {scores.max():.6f}")
            print(f"  Mean score: {scores.mean():.6f}")
            print(f"  Median score: {np.median(scores):.6f}")
            
            # Count detections at different thresholds
            thresholds = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
            print(f"\nDetections at different thresholds:")
            for thresh in thresholds:
                count = np.sum(scores >= thresh)
                print(f"  Threshold {thresh:.2f}: {count} detections")
            
            # Analyze top detections
            if len(scores) > 0:
                top_indices = np.argsort(scores)[-10:][::-1]  # Top 10
                print(f"\nTop 10 detections:")
                for i, idx in enumerate(top_indices):
                    if idx < len(scores):
                        print(f"  {i+1}. Score: {scores[idx]:.6f}, "
                              f"Label: {labels[idx]}, "
                              f"Box: [{boxes[idx][0]:.2f}, {boxes[idx][1]:.2f}, "
                              f"{boxes[idx][2]:.2f}, {boxes[idx][3]:.2f}]")
            
            # Check for common issues
            self._check_common_issues(boxes, scores, labels, original_size)
        else:
            print("⚠️  Unexpected number of outputs. Expected at least 3 (boxes, scores, labels)")
    
    def _check_common_issues(self, boxes, scores, labels, original_size):
        """Check for common issues that cause 0 mAP"""
        print(f"\n{'-'*40}")
        print("COMMON ISSUES CHECK")
        print(f"{'-'*40}")
        
        issues_found = []
        
        # Issue 1: All scores are too low
        max_score = scores.max()
        if max_score < 0.01:
            issues_found.append(f"❌ Maximum confidence score is very low: {max_score:.6f}")
        elif max_score < 0.1:
            issues_found.append(f"⚠️  Maximum confidence score is low: {max_score:.6f}")
        else:
            print(f"✓ Confidence scores look reasonable (max: {max_score:.3f})")
        
        # Issue 2: Boxes are in wrong coordinate system
        box_ranges = {
            'x_min': boxes[:, 0].min(),
            'y_min': boxes[:, 1].min(), 
            'x_max': boxes[:, 2].max(),
            'y_max': boxes[:, 3].max()
        }
        
        print(f"\nBounding box coordinate ranges:")
        for coord, value in box_ranges.items():
            print(f"  {coord}: {value:.2f}")
        
        # Check if boxes are normalized [0,1] or in pixel coordinates
        if box_ranges['x_max'] <= 1.0 and box_ranges['y_max'] <= 1.0:
            print("✓ Boxes appear to be in normalized coordinates [0,1]")
        elif box_ranges['x_max'] <= original_size[0] and box_ranges['y_max'] <= original_size[1]:
            print("✓ Boxes appear to be in pixel coordinates")
        else:
            issues_found.append(f"❌ Bounding boxes have unusual coordinate ranges")
        
        # Issue 3: Invalid boxes
        invalid_boxes = 0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
        
        if invalid_boxes > 0:
            issues_found.append(f"❌ Found {invalid_boxes} invalid boxes (x2<=x1 or y2<=y1)")
        else:
            print("✓ All boxes have valid coordinates")
        
        # Issue 4: Label range
        unique_labels = np.unique(labels)
        print(f"\nUnique labels found: {unique_labels[:10]}...")  # Show first 10
        if len(unique_labels) > 100:
            issues_found.append(f"⚠️  Very high number of unique labels: {len(unique_labels)}")
        
        # Issue 5: NaN or infinite values
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            issues_found.append("❌ Found NaN or infinite values in scores")
        if np.any(np.isnan(boxes)) or np.any(np.isinf(boxes)):
            issues_found.append("❌ Found NaN or infinite values in boxes")
        
        # Summary
        if issues_found:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue in issues_found:
                print(f"   {issue}")
        else:
            print(f"\n✅ No obvious issues found in model outputs")
    
    def test_with_different_thresholds(self, image_path):
        """Test detection with very low thresholds"""
        print(f"\n{'='*60}")
        print(f"TESTING WITH DIFFERENT THRESHOLDS")
        print(f"{'='*60}")
        
        # Get outputs
        image_array, original_image, original_size = self.preprocess_image(image_path)
        outputs = self.onnx_session.run(None, {self.input_name: image_array})
        
        if len(outputs) < 3:
            print("❌ Cannot test thresholds - unexpected output format")
            return
        
        boxes = outputs[0][0] if len(outputs[0].shape) > 2 else outputs[0]
        scores = outputs[1][0] if len(outputs[1].shape) > 1 else outputs[1]
        labels = outputs[2][0] if len(outputs[2].shape) > 1 else outputs[2]
        
        # Test various thresholds
        test_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        
        for threshold in test_thresholds:
            valid_mask = scores >= threshold
            valid_count = np.sum(valid_mask)
            
            print(f"Threshold {threshold:5.3f}: {valid_count:4d} detections", end="")
            
            if valid_count > 0:
                valid_scores = scores[valid_mask]
                print(f" (confidence range: {valid_scores.min():.3f} - {valid_scores.max():.3f})")
                
                # Show a few examples at this threshold
                if valid_count <= 5:
                    valid_boxes = boxes[valid_mask]
                    valid_labels = labels[valid_mask]
                    for i in range(valid_count):
                        print(f"    Detection {i+1}: score={valid_scores[i]:.3f}, "
                              f"label={valid_labels[i]}, "
                              f"box=[{valid_boxes[i][0]:.1f}, {valid_boxes[i][1]:.1f}, "
                              f"{valid_boxes[i][2]:.1f}, {valid_boxes[i][3]:.1f}]")
            else:
                print("")
    
    def create_debug_visualization(self, image_path, output_path, threshold=0.01):
        """Create visualization with very low threshold to see what model detects"""
        print(f"\n{'='*60}")
        print(f"CREATING DEBUG VISUALIZATION")
        print(f"{'='*60}")
        
        # Get outputs
        image_array, original_image, original_size = self.preprocess_image(image_path)
        outputs = self.onnx_session.run(None, {self.input_name: image_array})
        
        if len(outputs) < 3:
            print("❌ Cannot create visualization - unexpected output format")
            return
        
        boxes = outputs[0][0] if len(outputs[0].shape) > 2 else outputs[0]
        scores = outputs[1][0] if len(outputs[1].shape) > 1 else outputs[1]
        labels = outputs[2][0] if len(outputs[2].shape) > 1 else outputs[2]
        
        # Filter by threshold
        valid_mask = scores >= threshold
        
        if not np.any(valid_mask):
            print(f"❌ No detections found at threshold {threshold}")
            return
        
        valid_boxes = boxes[valid_mask]
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        
        print(f"✓ Found {len(valid_boxes)} detections at threshold {threshold}")
        
        # Create visualization
        draw = ImageDraw.Draw(original_image)
        
        # Scale boxes to original image size if needed
        scale_x = original_size[0] / 768  # Assuming input size 768
        scale_y = original_size[1] / 768
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        
        for i, (box, score, label) in enumerate(zip(valid_boxes, valid_scores, valid_labels)):
            x1, y1, x2, y2 = box
            
            # Scale coordinates if boxes are in normalized coordinates
            if x2 <= 1.0 and y2 <= 1.0:
                x1 *= original_size[0]
                y1 *= original_size[1]
                x2 *= original_size[0]
                y2 *= original_size[1]
            elif x2 <= 768 and y2 <= 768:  # Model input size coordinates
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
            
            color = colors[int(label) % len(colors)]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label
            label_text = f"L{int(label)}:{score:.3f}"
            draw.text((x1, y1-20), label_text, fill=color)
        
        # Save visualization
        original_image.save(output_path)
        print(f"✓ Debug visualization saved to: {output_path}")
    
    def run_comprehensive_debug(self, test_image_path):
        """Run all debugging tests"""
        print(f"🔍 COMPREHENSIVE ONNX MODEL DEBUG")
        print(f"Model: {self.onnx_model_path}")
        print(f"Test image: {test_image_path}")
        
        # 1. Analyze raw outputs
        outputs = self.analyze_onnx_outputs(test_image_path)
        
        # 2. Test different thresholds
        self.test_with_different_thresholds(test_image_path)
        
        # 3. Create debug visualization
        debug_viz_path = "debug_visualization.jpg"
        self.create_debug_visualization(test_image_path, debug_viz_path, threshold=0.001)
        
        # 4. Provide recommendations
        self._provide_recommendations()
    
    def _provide_recommendations(self):
        """Provide recommendations based on debugging results"""
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS")
        print(f"{'='*60}")
        
        print("Based on the debugging results, here are potential fixes:")
        print("")
        print("1. 🔧 MODEL CONVERSION ISSUES:")
        print("   - Re-export ONNX with different settings")
        print("   - Try different ONNX opset versions (11, 12, 13)")
        print("   - Include/exclude post-processing in ONNX model")
        print("")
        print("2. 🔧 POST-PROCESSING ISSUES:")
        print("   - Check coordinate system (normalized vs pixel)")
        print("   - Verify NMS is working correctly")
        print("   - Check class ID mapping")
        print("")
        print("3. 🔧 THRESHOLD ISSUES:")
        print("   - Use very low threshold (0.001) for evaluation")
        print("   - Check if sigmoid is applied to classification scores")
        print("")
        print("4. 🔧 DEBUGGING STEPS:")
        print("   - Compare ONNX vs PyTorch outputs on same image")
        print("   - Test with single image first")
        print("   - Verify preprocessing is identical")
        print("")
        print("Run this script with different test images to identify patterns!")

def main():
    parser = argparse.ArgumentParser(description='Debug ONNX EfficientDet Model')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--test_image', type=str, required=True, help='Path to test image')
    parser.add_argument('--pytorch_model', type=str, default=None, help='Path to original PyTorch model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx_model):
        print(f"❌ ONNX model not found: {args.onnx_model}")
        return 1
    
    if not os.path.exists(args.test_image):
        print(f"❌ Test image not found: {args.test_image}")
        return 1
    
    try:
        debugger = ONNXModelDebugger(args.onnx_model, args.pytorch_model)
        debugger.run_comprehensive_debug(args.test_image)
        
        print(f"\n🎉 Debugging completed!")
        print(f"Check 'debug_visualization.jpg' to see what the model detects")
        
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())