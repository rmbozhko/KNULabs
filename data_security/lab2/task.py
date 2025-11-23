"""Task implementation for ArTaxOr dataset with Flower FL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
from typing import Tuple
from pathlib import Path

# Global variable to store dataset
fds = None


class ArTaxOrDataset(Dataset):
    """Custom Dataset for ArTaxOr arthropod images."""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Path to the dataset directory containing images and annotations
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # ArTaxOr classes (7 arthropod orders)
        self.classes = [
            'Araneae',      # Spiders
            'Coleoptera',   # Beetles
            'Diptera',      # Flies
            'Hemiptera',    # True bugs
            'Hymenoptera',  # Wasps, bees, ants
            'Lepidoptera',  # Butterflies, moths
            'Odonata'       # Dragonflies, damselflies
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load images and labels from directory structure or annotation file."""
        # Assuming directory structure: data_dir/{class_name}/{image_files}
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                for img_path in class_dir.glob('*.jpeg'):
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {"img": image, "label": label}


def load_data(partition_id: int, num_partitions: int, data_dir: str = "./artaxor_data") -> Tuple[DataLoader, DataLoader]:
    """Load partitioned ArTaxOr data.
    
    Args:
        partition_id: ID of the partition to load (0 to num_partitions-1)
        num_partitions: Total number of partitions
        data_dir: Path to the ArTaxOr dataset directory
    
    Returns:
        Tuple of (trainloader, testloader)
    """
    # Define transforms (resize to 224x224, normalize)
    pytorch_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = ArTaxOrDataset(data_dir, transform=pytorch_transforms)
    
    # Calculate partition size
    total_size = len(full_dataset)
    partition_size = total_size // num_partitions
    
    # Calculate start and end indices for this partition
    start_idx = partition_id * partition_size
    if partition_id == num_partitions - 1:
        end_idx = total_size  # Last partition gets remaining data
    else:
        end_idx = start_idx + partition_size
    
    # Create partition subset
    partition_indices = list(range(start_idx, end_idx))
    partition_dataset = torch.utils.data.Subset(full_dataset, partition_indices)
    
    # Split partition into 80% train, 20% test
    train_size = int(0.8 * len(partition_dataset))
    test_size = len(partition_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        partition_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    
    return trainloader, testloader


class Net(nn.Module):
    """CNN model for ArTaxOr classification (7 classes)."""
    
    def __init__(self, num_classes: int = 7):
        super(Net, self).__init__()
        # Input: 224x224x3 images
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs: int, lr: float, device: torch.device):
    """Train the model on the training set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    running_loss = 0.0
    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss


def test(net, testloader, device: torch.device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy