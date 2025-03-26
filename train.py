import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNISTModel
import os

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data transformations with augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=15,  # Random rotation +/- 15 degrees
            translate=(0.1, 0.1),  # Random translation up to 10%
            scale=(0.9, 1.1),  # Random scaling between 90% and 110%
            shear=10  # Random shearing up to 10 degrees
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Brightness/contrast variation
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transformations should be just normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets with augmented training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096)
    
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.6f}')
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Test phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 'models/mnist_model.pth')
    
    # Plot metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == '__main__':
    train_model()
