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
        transforms.Resize((224, 224)),  # Resize images for MobileNet input
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data
data_dir = 'data/'
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'validation': datasets.ImageFolder(os.path.join(data_dir, 'validation'), data_transforms['validation']),
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'validation': DataLoader(image_datasets['validation'], batch_size=32, shuffle=False),
}

# Get the class names (Arial, Courier New, Tahoma, Amiri, Andalus)
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

# Load MobileNetV2 pretrained model
model = models.mobilenet_v2(pretrained=True)
# Modify the classifier to fit our five-class problem
model.classifier[1] = nn.Linear(model.last_channel, 5)  # Change the output layer to 5 classes
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(image_datasets['train'])
        print(f'Training Loss: {epoch_loss:.4f}')
        
    print("Training complete!")

# Training the model
train_model(model, criterion, optimizer)

# Evaluate model and get predictions
model.eval()
all_preds = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for inputs, labels in dataloaders['validation']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_filenames.extend([image_datasets['validation'].imgs[i][0] for i in range(len(inputs))])

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Show some misclassified images
misclassified_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
num_misclassified = min(len(misclassified_indices), 6)  # Show up to 6 examples

fig, axes = plt.subplots(1, num_misclassified, figsize=(12, 8))

for i, idx in enumerate(misclassified_indices[:num_misclassified]):
    img = Image.open(all_filenames[idx])
    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {class_names[all_preds[idx]]}\nTrue: {class_names[all_labels[idx]]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
