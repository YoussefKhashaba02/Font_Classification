import argparse
import torch
import torchvision.models as models
import onnxruntime as ort

def export_to_onnx(model_path, onnx_model_path):
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2) #Adjust for Number of classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(model, dummy_input, onnx_model_path, 
                      input_names=['input'], output_names=['output'], 
                      opset_version=11)

    print(f"Model exported to {onnx_model_path}")

    ort_session = ort.InferenceSession(onnx_model_path)
    print(f"ONNX model loaded from {onnx_model_path} successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a PyTorch model to ONNX format.")
    parser.add_argument('model_path', type=str, help="Path to the PyTorch model file (.pth)")
    parser.add_argument('onnx_model_path', type=str, help="Path where the ONNX model will be saved")

    args = parser.parse_args()

    export_to_onnx(args.model_path, args.onnx_model_path)
