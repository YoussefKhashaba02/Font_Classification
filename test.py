import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from torchvision.models import resnet18
import numpy as np
import cv2

# Load the ResNet18 model
model_path = 'resnet18_font_classification_new.pth'  # Replace with your actual model file path
model = resnet18(pretrained=False)  # Initialize ResNet18 without pretrained weights
model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Modify the last layer to match the number of classes
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()  # Set the model to evaluation mode

# Store the gradients and activations of the last layer
gradients = None
activations = None

def save_gradients(grad):
    global gradients
    gradients = grad

def save_activations(act):
    global activations
    activations = act

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Denormalize function to convert the image back to a displayable format
def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    image = tensor.squeeze(0)  # Remove batch dimension
    image = image * std + mean  # Denormalize
    image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = torch.clamp(image, 0, 1)  # Ensure values are within [0, 1]
    
    # Convert to numpy and scale to [0, 255]
    image = (image.numpy() * 255).astype(np.uint8)
    return image

# Function to generate the Grad-CAM heatmap
def generate_gradcam(image, class_index):
    global gradients, activations
    model.eval()
    
    # Register hooks to capture gradients and activations
    final_conv_layer = model.layer4[-1]  # Use the last convolutional layer of ResNet18
    final_conv_layer.register_full_backward_hook(lambda m, grad_input, grad_output: save_gradients(grad_output[0]))
    final_conv_layer.register_forward_hook(lambda m, input, output: save_activations(output))

    # Forward pass
    output = model(image)
    
    # Zero gradients and perform backward pass
    model.zero_grad()
    class_loss = output[0][class_index]
    class_loss.backward()
    
    # Compute the weights
    weights = torch.mean(gradients, dim=(0, 2, 3))  # Global average pooling
    cam = torch.zeros(activations.shape[2:], dtype=torch.float32)

    # Create the heatmap
    for i, w in enumerate(weights):
        if i < activations.shape[1]:  # Check if the index is within bounds
            cam += w * activations[0, i, :, :]
    
    # ReLU and normalize
    cam = torch.clamp(cam, min=0)
    cam /= torch.max(cam)

    return cam.detach().numpy()  # Detach the tensor before converting to NumPy

# Define prediction function using PyTorch
def classify_font(image):
    image = Image.fromarray(image)  # Convert numpy array to PIL image
    preprocessed_image = preprocess_image(image)

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(preprocessed_image)
    
    # Get predicted class
    font_classes = ['Amiri', 'Andalus', 'Arial', 'Courier New', 'Tahoma']  # Adjust according to your classes
    softmax = torch.nn.functional.softmax(output, dim=1)  # Apply softmax to get probabilities
    top_probs, top_preds = torch.topk(softmax, 5)  # Get top 5 predictions

    # Create a result dictionary that maps the original class indices correctly
    result = {font_classes[pred]: float(top_probs[0][i]) for i, pred in enumerate(top_preds[0])}
    
    # Print the prediction to the console
    print("Model Prediction:", result)

    # Generate Grad-CAM heatmap for the top predicted class
    class_index = top_preds[0][0].item()  # Get the index of the top prediction
    cam = generate_gradcam(preprocessed_image, class_index)

    # Overlay the heatmap on the original image
    original_image = np.array(image.resize((224, 224)))  # Resize to match the model input size
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (224, 224))  # Resize to original image size
    overlayed_image = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # Denormalize the preprocessed image for display
    preprocessed_image_display = denormalize_image(preprocessed_image)
    
    # Return the overlayed image and the prediction result
    return overlayed_image, result

# Set up Gradio UI
interface = gr.Interface(
    fn=classify_font,
    inputs=gr.Image(),  # Accept an image input
    outputs=[gr.Image(label="Overlayed Image"), gr.Label(num_top_classes=5)],  # Show overlayed image and top 5 classes with their probabilities
    title="Font Classification with Grad-CAM",
    description="Upload an image of text to classify whether the font is Arial, Courier New, Tahoma, Amiri, or Andalus."
)

# Launch the Gradio interface
interface.launch()
