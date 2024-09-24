import onnx
import onnxruntime as ort
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

# Load the ONNX model
onnx_model_path = 'fonts_model'
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
ort_session = ort.InferenceSession(onnx_model_path)

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

# Define prediction function using ONNX Runtime
def classify_font(image):
    image = Image.fromarray(image)  # Convert numpy array to PIL image
    input_data = preprocess_image(image)
    
    # Perform inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Get predicted class
    font_classes = ['Arial','Courier New']
    output = torch.tensor(ort_outs[0])
    _, pred = torch.max(output, 1)
    return font_classes[pred.item()]

# Set up Gradio UI
interface = gr.Interface(
    fn=classify_font,
    inputs=gr.Image(),  # Removed shape argument
    outputs=gr.Label(num_top_classes=2),
    title="Font Classification",
    description="Upload an image of text to classify if the font is Courier New or Arial."
)

# Launch the Gradio interface
interface.launch()
