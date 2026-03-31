import torch
from torchvision import transforms, models, datasets  # FIXED: Added datasets import
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define the same transforms used for validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load your class names
class_names = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']

# 3. Load the saved model
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, len(class_names))

# Load the model weights from the correct path
model_path = './results/insect_mobilenet_v3_final.pth'  # Updated path
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    # Fallback to old path if new one doesn't exist
    model.load_state_dict(torch.load('insect_mobilenet_v3.pth', map_location=device))

model = model.to(device)
model.eval()

# 4. Function to predict an image with visualization
def predict_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence = probabilities[predicted[0]].item()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show the image
    ax1.imshow(np.array(image))
    ax1.set_title(f'Prediction: {class_names[predicted[0]]}\nConfidence: {confidence:.2%}')
    ax1.axis('off')
    
    # Create confidence bar chart
    colors = ['lightblue' for _ in range(len(class_names))]
    colors[predicted[0]] = 'green'  # Highlight the predicted class
    
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities.cpu().numpy(), color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Confidence')
    ax2.set_title('Class Confidence Scores')
    ax2.set_xlim(0, 1)
    
    # Add confidence values to bars
    for i, v in enumerate(probabilities.cpu().numpy()):
        ax2.text(v + 0.01, i, f'{v:.2%}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return class_names[predicted[0]], confidence, predicted[0].item()

# 5. Function to generate confusion matrix
def plot_confusion_matrix():
    # Load validation dataset
    val_dataset = datasets.ImageFolder('./insects-data/val', transform=transform)  # FIXED: transform=transform (not transform-transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print("Generating predictions for confusion matrix...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Calculate overall accuracy
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Overall Validation Accuracy: {accuracy:.2%}")
    
    return cm

# 6. Function to plot training history
def plot_training_history():
    history_file = './results/training_history.json'  # Updated path
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Epochs', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No training history found. Run the modified training code first.")

# 7. Function to show sample predictions - FIXED VERSION
def show_sample_predictions(num_samples=5):
    # Load validation dataset
    val_dataset = datasets.ImageFolder('./insects-data/val', transform=transform)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(len(val_dataset))
        image, true_label = val_dataset[idx]
        
        # Get the original image path for display
        image_path, _ = val_dataset.samples[idx]
        original_image = Image.open(image_path).convert('RGB')
        
        # Predict
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities[predicted[0]].item()
        
        # Plot
        axes[i].imshow(np.array(original_image))
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[predicted[0]]}\nConf: {confidence:.2%}', fontsize=10)
        axes[i].axis('off')
        
        # Color code: green if correct, red if wrong
        color = 'green' if predicted[0] == true_label else 'red'
        for spine in axes[i].spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    print("MobileNetV3 Insect Classifier Analysis")
    print("="*50)
    
    # Option 1: Single image prediction
    image_path = input("Enter the path to your insect image (or press Enter for analysis only): ").strip().strip('"')
    
    if image_path and os.path.exists(image_path):
        predicted_class, confidence, pred_idx = predict_image(image_path)
        print(f"\nPredicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    
    # Option 2: Comprehensive analysis
    print("\nGenerating comprehensive model analysis...")
    
    # Plot training history
    plot_training_history()
    
    # Generate confusion matrix
    cm = plot_confusion_matrix()
    
    # Show sample predictions
    print("\nShowing random sample predictions from validation set:")
    show_sample_predictions(num_samples=5)
    
    print("\nAnalysis complete! Review the plots to understand your model's performance.")