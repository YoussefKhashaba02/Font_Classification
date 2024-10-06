import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (data augmentation and normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images for model input
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,hue=0.25),  # Adjust contrast and brightness 
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)  # Optionally add a slight blur
            ], p=0.5),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data
data_dir = 'data'
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'validation': datasets.ImageFolder(os.path.join(data_dir, 'validation'), data_transforms['validation']),
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validation': DataLoader(image_datasets['validation'], batch_size=32, shuffle=False),
}

# Get the class names
class_names = image_datasets['train'].classes

# Function to check for common data between training and validation sets
def check_common_data(train_dataset, validation_dataset):
    # Get the list of file paths for both datasets
    train_files = set([img_path for img_path, _ in train_dataset.imgs])
    validation_files = set([img_path for img_path, _ in validation_dataset.imgs])

    # Find the common files
    common_files = train_files.intersection(validation_files)

    if len(common_files) > 0:
        print(f"Warning: {len(common_files)} common images found between train and validation sets.")
    else:
        print("No common images found between train and validation sets.")
        
# Check for common data before training
check_common_data(image_datasets['train'], image_datasets['validation'])

# Function to train the model
def train_model(model, model_name, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} ({model_name})')
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # Accuracy
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
        
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = correct_train / total_train
        print(f'{model_name} Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
        
    print(f"{model_name} training complete!")
    
    # Save the trained model
    model_path = f'{model_name}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
# Function to evaluate the model
def evaluate_model(model, model_name, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    correct_val = 0
    total_val = 0
    running_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend([image_datasets['validation'].imgs[i][0] for i in range(len(inputs))])

            correct_val += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)

    # Calculate validation loss and accuracy
    val_loss = running_val_loss / len(image_datasets['validation'])
    val_acc = correct_val / total_val
    print(f'{model_name} Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.4f}')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()

    # Show and save misclassified images
    misclassified_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
    num_misclassified = min(len(misclassified_indices), 6)  # Show up to 6 examples
    fig, axes = plt.subplots(1, num_misclassified, figsize=(12, 8))

    for i, idx in enumerate(misclassified_indices[:num_misclassified]):
        img = Image.open(all_filenames[idx])
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {class_names[all_preds[idx]]}\nTrue: {class_names[all_labels[idx]]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{model_name}_misclassified_images.png")
    plt.show()

# Loss function
criterion = nn.CrossEntropyLoss()

# Load and modify MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 5)  # 5 classes
mobilenet = mobilenet.to(device)
optimizer_mobilenet = optim.Adam(mobilenet.parameters(), lr=0.001)

# Train and evaluate MobileNetV2
train_model(mobilenet, 'mobilenet_v2_font_classification', criterion, optimizer_mobilenet)
evaluate_model(mobilenet, 'mobilenet_v2_font_classification', criterion)

# Load and modify ResNet18
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 5)  # 5 classes
resnet = resnet.to(device)
optimizer_resnet = optim.Adam(resnet.parameters(), lr=0.001)

# Train and evaluate ResNet18
train_model(resnet, 'resnet18_font_classification', criterion, optimizer_resnet)
evaluate_model(resnet, 'resnet18_font_classification', criterion)
