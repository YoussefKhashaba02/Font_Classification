import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import random
import time
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
import copy

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
        """Process the raw input data"""
        self.target_height = 32
        self.target_width = 804
        if not isinstance(inputs, list):
            inputs = [inputs]
        data_batch = []
        for item in inputs:
            if isinstance(item, str):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, PIL.Image.Image):
                img = np.array(item.convert('RGB'))
            elif isinstance(item, np.ndarray):
                img = item
            else:
                raise TypeError(
                    f'Inputs should be either (a list of) str, PIL.Image, np.array, but got {type(item)}'
                )
            img = self.keepratio_resize(img)
            img = torch.FloatTensor(img)
            if True:
                chunk_img = [img[:, (300 - 48) * i: (300 - 48) * i + 300] for i in range(3)]
                merge_img = torch.cat(chunk_img, 0)
                data = merge_img.permute(2, 0, 1)
            else:
                data = img.view(1, self.target_height, self.target_width, 3) / 255.
                data = data.permute(0, 3, 1, 2)
            data_batch.append(data)
        data_batch = torch.cat(data_batch, 0)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(data / 255)
        return pil_image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_file)
                    if img_file.endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(set(self.labels))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[self.labels[idx]]

        ocr_preprocessor = OCRRecognitionPreprocessor()
        image = ocr_preprocessor(image)

        if self.transform:
            image = self.transform(image)

        return np.array(image), label

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

train_dir = 'data/train'
validation_dir = 'data/validation'

image_datasets = {
    'train': CustomDataset(train_dir, transform=data_transforms['train']),
    'validation': CustomDataset(validation_dir, transform=data_transforms['validation'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'validation': DataLoader(image_datasets['validation'], batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].class_to_idx.keys()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_samples(dataloader, num_samples=5):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        img = images[i]
        #img = denormalize(img, mean, std)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.show()

if __name__ == '__main__':
    show_samples(dataloaders['train'], num_samples=5)

    criterion = nn.CrossEntropyLoss()

    # Train and save MobileNet
    mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet_model.classifier[1] = nn.Linear(mobilenet_model.last_channel, len(class_names))
    mobilenet_model = mobilenet_model.to(device)

    optimizer_mobilenet = optim.SGD(mobilenet_model.parameters(), lr=0.001, momentum=0.9)

    mobilenet_model = train_model(mobilenet_model, criterion, optimizer_mobilenet, num_epochs=25)
    torch.save(mobilenet_model.state_dict(), 'mobilenet_model.pth')

    # Train and save ResNet18
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, len(class_names))
    resnet_model = resnet_model.to(device)

    optimizer_resnet = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

    resnet_model = train_model(resnet_model, criterion, optimizer_resnet, num_epochs=25)
    torch.save(resnet_model.state_dict(), 'resnet_model.pth')
