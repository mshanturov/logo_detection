import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import cv2
import xml.etree.ElementTree as ET
import os
import yaml
import shutil

def load_data(data_dir, transform):
    # Loads the data from the specified directory, applying the given transform.
    # Returns two Subset objects: one for the training set and one for the validation set.
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.targets,
        random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size):
    # Creates DataLoaders for the training and validation sets.
    # DataLoader provides efficient data loading and shuffling.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_transforms(classifier=True):
    # Defines the image transforms for either the classifier or the object detector.
    if classifier:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: # for YOLO
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform

def parse_annotation(xml_path):
    # Parses the XML annotation file and extracts the bounding box information.
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, name]) # Include name for multi-class
    return boxes


