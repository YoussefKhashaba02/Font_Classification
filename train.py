import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import copy
import time
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt

class OCRRecognitionPreprocessor:
    def keepratio_resize(self, img):
        target_height = 32
        target_width = 804
        try:
            if img is None or len(img.shape) < 2:
                raise ValueError("Invalid image input")

            cur_ratio = img.shape[1] / float(img.shape[0])
            mask_height = target_height
            mask_width = target_width

            if cur_ratio > float(target_width) / target_height:
                cur_target_height = target_height
                cur_target_width = target_width
            else:
                cur_target_height = target_height
                cur_target_width = int(target_height * cur_ratio)

            img = cv2.resize(img, (cur_target_width, cur_target_height))

            # Handle both grayscale and color images
            if len(img.shape) == 2:
                mask = np.zeros([mask_height, mask_width], dtype=img.dtype)
                mask[:img.shape[0], :img.shape[1]] = img
            else:
                mask = np.zeros([mask_height, mask_width, img.shape[2]], dtype=img.dtype)
                mask[:img.shape[0], :img.shape[1], :] = img

            return mask
        except Exception as e:
            print(f"Error in keepratio_resize: {e}")
            return None

    def __call__(self, inputs):
        """process the raw input data
        Args:
            inputs:
                - A string containing an HTTP link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL or opencv directly
        Returns:
            outputs: the preprocessed image
        """
        self.target_height = 32
        self.target_width = 804
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            # Load and convert image based on input type
            if isinstance(item, str):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, PIL.Image.Image):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, np.ndarray):
                img = item
            else:
                raise TypeError(
                    f'inputs should be either (a list of) str, PIL.Image, np.array, but got {type(item)}'
                )
            img = self.keepratio_resize(img)
            img = torch.FloatTensor(img)
            if True:
                chunk_img = [img[:, (300 - 48) * i : (300 - 48) * i + 300] for i in range(3)]
                merge_img = torch.cat(chunk_img, 0)
                data = merge_img.permute(2, 0, 1)
            else:
                data = img.view(1, self.target_height, self.target_width,
                                3) / 255.
                data = data.permute(0, 3, 1, 2)
            data_batch.append(data)
        data_batch = torch.cat(data_batch, 0)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(data/255)
        return pil_image
# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Iterate through all the classes in the root directory
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if img_file.endswith(('png', 'jpg', 'jpeg')):  # Add other image formats if needed
                        self.image_paths.append(img_path)
                        self.labels.append(label)  # Store the label based on the folder name

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(set(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.labels[idx]]

        # Create an instance of OCRRecognitionPreprocessor
        ocr_preprocessor = OCRRecognitionPreprocessor()

        # Preprocess the image using the OCRRecognitionPreprocessor
        image = ocr_preprocessor(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomInvert(p=0.4),
        transforms.RandomApply([transforms.RandomResizedCrop((200, 200)), transforms.Resize((224, 224))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.Pad(padding=random.randint(1, 20), fill=(0, 0, 0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Setup directories
train_dir = 'data/train'
validation_dir = 'data/validation'

# Load datasets using the custom dataset
image_datasets = {
    'train': CustomDataset(train_dir, transform=data_transforms['train']),
    'validation': CustomDataset(validation_dir, transform=data_transforms['validation'])
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'validation': DataLoader(image_datasets['validation'], batch_size=32, shuffle=False, num_workers=4)
}

# Get the dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}

# Get the class names
class_names = image_datasets['train'].class_to_idx.keys()

# Set the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to train and evaluate the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_samples(dataloader, num_samples=5):
    """Displays a number of sample images after preprocessing.

    Args:
        dataloader: The dataloader from which to fetch images.
        num_samples: The number of images to display.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))

    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        img = denormalize(images[i].clone(), mean, std)
        img = img.permute(1, 2, 0).numpy()  # Convert to numpy array for display
        img = (img * 255).astype(np.uint8)  # Rescale to [0, 255]
        plt.imshow(img)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.show()




if __name__ == '__main__':
    # Show sample images before training
    show_samples(dataloaders['train'], num_samples=5)

    # Set up the criterion and learning rate scheduler
    criterion = nn.CrossEntropyLoss()

    # Train and save MobileNet
    mobilenet_model = models.mobilenet_v2(pretrained=True)
    mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, len(class_names))
    mobilenet_model = mobilenet_model.to(device)

    optimizer_mobilenet = optim.SGD(mobilenet_model.parameters(), lr=0.001, momentum=0.9)
    scheduler_mobilenet = optim.lr_scheduler.StepLR(optimizer_mobilenet, step_size=7, gamma=0.1)

    mobilenet_model = train_model(mobilenet_model, criterion, optimizer_mobilenet, scheduler_mobilenet, num_epochs=25)
    torch.save(mobilenet_model.state_dict(), 'mobilenet_v2.pth')

    # Train and save ResNet18
    resnet_model = models.resnet18(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model = resnet_model.to(device)

    optimizer_resnet = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
    scheduler_resnet = optim.lr_scheduler.StepLR(optimizer_resnet, step_size=7, gamma=0.1)

    resnet_model = train_model(resnet_model, criterion, optimizer_resnet, scheduler_resnet, num_epochs=25)
    torch.save(resnet_model.state_dict(), 'resnet18.pth')

    print("Training complete and models saved!")
