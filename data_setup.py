import os
import shutil
import random
from pathlib import Path

# Directories containing the images
source_dir_arial = 'data/arial'
source_dir_couriernew = 'data/couriernew'

# Target directories for train and validation sets
train_dir = 'data/train'
validation_dir = 'data/validation'

# Create the necessary folders
os.makedirs(os.path.join(train_dir, 'arial'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'couriernew'), exist_ok=True)
os.makedirs(os.path.join(validation_dir, 'arial'), exist_ok=True)
os.makedirs(os.path.join(validation_dir, 'couriernew'), exist_ok=True)

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

# Split Arial images
split_data(source_dir_arial, os.path.join(train_dir, 'arial'), os.path.join(validation_dir, 'arial'))

# Split Courier New images
split_data(source_dir_couriernew, os.path.join(train_dir, 'couriernew'), os.path.join(validation_dir, 'couriernew'))

print("Data split complete!")
