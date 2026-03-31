import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load dataset
data_dir = './insects-data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                  data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Class names: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")

# Create results directory
os.makedirs('./results', exist_ok=True)

# Load pre-trained MobileNetV3
model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function with history tracking
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialize history tracking
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'learning_rate': [], 'epoch_times': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(float(epoch_loss))
                history['train_acc'].append(float(epoch_acc))
                history['learning_rate'].append(float(scheduler.get_last_lr()[0]))
            else:
                history['val_loss'].append(float(epoch_loss))
                history['val_acc'].append(float(epoch_acc))

        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(float(epoch_time))
        print(f'Epoch time: {epoch_time:.2f}s\n')

        # Save best model
        if history['val_acc'][-1] > best_acc:
            best_acc = history['val_acc'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Function to generate and save confusion matrix
def save_confusion_matrix(model, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Generating confusion matrix...")
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = model(inputs)
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
    plt.savefig('./results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix as numpy array
    np.save('./results/confusion_matrix.npy', cm)
    
    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open('./results/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save readable classification report
    with open('./results/classification_report.txt', 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    return cm, report

# Function to save training plots
def save_training_plots(history):
    # Loss plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning rate plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['learning_rate'], linewidth=2, color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig('./results/learning_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

# Function to save final results summary
def save_results_summary(history, cm, report, class_names):
    summary = {
        'final_training_accuracy': history['train_acc'][-1],
        'final_validation_accuracy': history['val_acc'][-1],
        'best_validation_accuracy': max(history['val_acc']),
        'final_training_loss': history['train_loss'][-1],
        'final_validation_loss': history['val_loss'][-1],
        'total_training_time_minutes': sum(history['epoch_times']) / 60,
        'average_epoch_time_seconds': np.mean(history['epoch_times']),
        'class_names': class_names,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {}
    }
    
    # Calculate per-class accuracy from confusion matrix
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, class_name in enumerate(class_names):
        summary['per_class_accuracy'][class_name] = float(class_accuracy[i])
    
    # Add classification report metrics
    for class_name in class_names:
        if class_name in report:
            summary[class_name] = report[class_name]
    
    with open('./results/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save readable summary
    with open('./results/training_summary.txt', 'w') as f:
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Final Training Accuracy: {summary['final_training_accuracy']:.4f}\n")
        f.write(f"Final Validation Accuracy: {summary['final_validation_accuracy']:.4f}\n")
        f.write(f"Best Validation Accuracy: {summary['best_validation_accuracy']:.4f}\n")
        f.write(f"Total Training Time: {summary['total_training_time_minutes']:.2f} minutes\n")
        f.write(f"Average Epoch Time: {summary['average_epoch_time_seconds']:.2f} seconds\n\n")
        
        f.write("PER-CLASS ACCURACY:\n")
        f.write("-" * 20 + "\n")
        for class_name, acc in summary['per_class_accuracy'].items():
            f.write(f"{class_name:15}: {acc:.4f}\n")
        
        f.write("\nDETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 35 + "\n")
        for class_name in class_names:
            if class_name in report:
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
                f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
                f.write(f"  F1-Score:  {report[class_name]['f1-score']:.4f}\n")
                f.write(f"  Support:   {report[class_name]['support']}\n")

# Main training execution
if __name__ == '__main__':
    print("Starting MobileNetV3 Training...")
    print("=" * 50)
    
    # Train the model
    num_epochs = 15
    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # Save the trained model
    torch.save(model.state_dict(), './results/insect_mobilenet_v3_final.pth')
    print("Model saved as './results/insect_mobilenet_v3_final.pth'")
    
    # Save training history
    with open('./results/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Generate and save confusion matrix
    cm, report = save_confusion_matrix(model, class_names)
    
    # Save training plots
    save_training_plots(history)
    
    # Save final results summary
    save_results_summary(history, cm, report, class_names)
    
    print("\n" + "=" * 50)
    print("ALL RESULTS SAVED IN './results/' FOLDER:")
    print("=" * 50)
    print("1. training_history.json - Complete training history")
    print("2. training_history.png - Loss and accuracy plots")
    print("3. learning_rate.png - Learning rate schedule")
    print("4. confusion_matrix.png - Visual confusion matrix")
    print("5. confusion_matrix.npy - Numerical confusion matrix")
    print("6. classification_report.json - Detailed metrics")
    print("7. classification_report.txt - Readable report")
    print("8. training_summary.json - Complete results summary")
    print("9. training_summary.txt - Readable summary")
    print("10. insect_mobilenet_v3_final.pth - Trained model weights")
    print("=" * 50)