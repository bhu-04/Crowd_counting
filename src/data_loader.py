import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import scipy.io as sio

import config # Import configuration settings

class ShanghaiTechADataset(Dataset):
    """Custom Dataset for ShanghaiTech Part A (Weakly Supervised - Count Level)"""
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.annot_dir = annot_dir
        self.transform = transform
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Load Annotation (.mat file) to get the ground-truth count
        annot_filename = 'GT_' + os.path.basename(img_path).replace('.jpg', '.mat')
        annot_path = os.path.join(self.annot_dir, annot_filename)
        
        try:
            mat = sio.loadmat(annot_path)
            # Total count is derived from the number of head annotations
            ground_truth_count = mat['image_info'][0,0]['num_heads'][0,0].item() 
        except Exception:
            # Placeholder for annotation issues
            ground_truth_count = 0 
            
        # Apply Data Augmentation/Transformations
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(ground_truth_count, dtype=torch.float32)

# Define Training Transformations (Includes Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomCrop(config.CROP_SIZE), # Simulates patch creation
    transforms.RandomHorizontalFlip(), 
    transforms.RandomGrayscale(p=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Test Transformations (Deterministic)
test_transform = transforms.Compose([
    transforms.Resize(config.IMAGE_RESIZE), 
    transforms.CenterCrop(config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders():
    """Initializes and returns the train and test data loaders."""
    train_data = ShanghaiTechADataset(config.TRAIN_IMG_DIR, config.TRAIN_ANNOT_DIR, transform=train_transform)
    test_data = ShanghaiTechADataset(config.TEST_IMG_DIR, config.TEST_ANNOT_DIR, transform=test_transform)
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader