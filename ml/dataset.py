import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .analysis import fft_analysis, noise_residual_analysis

class RealFakeDataset(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, fixed_size=224):
        """
        Args:
            root_dir (str): Path to data directory (e.g. 'data/train') containing 'real' and 'fake' subdirs.
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            fixed_size (int): Size to resize images to.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.fixed_size = fixed_size
        self.image_paths = []
        self.labels = [] # 0: Real, 1: Fake
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        # Load Real
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                self.image_paths.append(os.path.join(real_dir, img_name))
                self.labels.append(0.0)
                
        # Load Fake
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                self.image_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(1.0)
                
        if not transform:
            if phase == 'train':
                self.transform = A.Compose([
                    A.Resize(fixed_size, fixed_size),
                    A.HorizontalFlip(p=0.5),
                    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2), # JPEG compression
                    A.GaussianBlur(p=0.1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(fixed_size, fixed_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load Image (RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Handle corrupt images safely
            return self.__getitem__((idx + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Generate Analysis Streams BEFORE normalization 
        # (FFT and Noise are sensitive to pixel values, but we might want them resized)
        # Let's resize first for consistency using a simple resize
        image_resized = cv2.resize(image, (self.fixed_size, self.fixed_size))
        
        # FFT
        fft_map = fft_analysis(image_resized) # Returns 2D array
        fft_tensor = torch.tensor(fft_map, dtype=torch.float32).unsqueeze(0) # (1, H, W)
        
        # Noise
        noise_map = noise_residual_analysis(image_resized) # Returns (H, W, 3) or similar
        noise_tensor = torch.tensor(noise_map, dtype=torch.float32).permute(2, 0, 1) # (3, H, W)
        
        # 2. Main Stream Augmentation & Normalization
        augmented = self.transform(image=image)['image']
        
        # Return dict or tuple
        return {
            'rgb': augmented,
            'fft': fft_tensor,
            'noise': noise_tensor,
            'label': torch.tensor(label, dtype=torch.float32)
        }
