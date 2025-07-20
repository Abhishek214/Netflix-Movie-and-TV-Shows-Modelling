import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image

def simple_efficientdet_inference(onnx_model_path, image_path):
    """
    Simple inference function for EfficientDet ONNX model
    """
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to model input size (typically 768x768 for D2)
    input_size = input_shape[2] if len(input_shape) == 4 else 768
    image_resized = image.resize((input_size, input_size), Image.BILINEAR)
    
    # Convert to numpy and normalize
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to CHW format and add batch dimension
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    # Run inference
    outputs = session.run(None, {input_name: image_array})
    
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    return outputs, original_size

# Example usage
if __name__ == "__main__":
    model_path = "efficientdet_d2.onnx"  # Update with your model path
    image_path = "test_image.jpg"        # Update with your image path
    
    try:
        outputs, original_size = simple_efficientdet_inference(model_path, image_path)
        print("Inference completed successfully!")
        print(f"Original image size: {original_size}")
        
        # Print first few detections if available
        if len(outputs) > 0 and len(outputs[0]) > 0:
            print("First few detections:")
            for i in range(min(5, len(outputs[0]))):
                print(f"Detection {i}: {outputs[0][i]}")
                
    except Exception as e:
        print(f"Error: {e}")