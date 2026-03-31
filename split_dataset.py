import os
import shutil
from sklearn.model_selection import train_test_split

# Path to your downloaded dataset (replace with your actual path)
data_dir = './insects-recognition'
# Paths for the new train and validation directories
train_dir = './insects-data/train'
val_dir = './insects-data/val'

# Create the new directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all class names from the original folder
class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Split ratio: 80% for training, 20% for validation
test_size = 0.2

for class_name in class_names:
    print(f"Processing class: {class_name}")
    
    # Path to the original class folder
    class_dir = os.path.join(data_dir, class_name)
    # Get all image files in the class folder
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    
    # Split the images into training and validation sets
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)
    
    # Create class subdirectories in train and val
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    # Copy training images
    for img in train_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copyfile(src, dst)
    
    # Copy validation images
    for img in val_images:
        src = os.path.join(class_dir, img)
        dst = os.path.join(val_dir, class_name, img)
        shutil.copyfile(src, dst)

print("Dataset splitting complete! Check the 'insects-data' folder.")