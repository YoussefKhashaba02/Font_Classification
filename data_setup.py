import os
import shutil
import random

# Directories containing the images for each font class
source_dirs = {
    'arial': 'data/arial',
    'couriernew': 'data/couriernew',
    'tahoma': 'data/tahoma',
    'amiri': 'data/amiri',
    'andalus': 'data/andalus'
}

# Target directories for train and validation sets
train_dir = 'data/train'
validation_dir = 'data/validation'

# Create necessary folders for each class in both train and validation sets
for font_class in source_dirs.keys():
    os.makedirs(os.path.join(train_dir, font_class), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, font_class), exist_ok=True)

def split_data(source_dir, train_dir, validation_dir, split_ratio=0.9):
    # Get all images from the source directory
    all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Shuffle the images to randomize the split
    random.shuffle(all_images)
    
    # Calculate the split index
    split_idx = int(len(all_images) * split_ratio)
    
    # Split the images
    train_images = all_images[:split_idx]
    validation_images = all_images[split_idx:]
    
    # Move the images to the appropriate folders
    for image in train_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(train_dir, image))
        
    for image in validation_images:
        shutil.copy(os.path.join(source_dir, image), os.path.join(validation_dir, image))

# Split data for each font class
for font_class, source_dir in source_dirs.items():
    split_data(source_dir, os.path.join(train_dir, font_class), os.path.join(validation_dir, font_class))

print("Data split complete!")
