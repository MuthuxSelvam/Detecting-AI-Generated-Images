import torch
import torch.nn as nn
import timm

class EnsembleDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # RGB Stream (EfficientNet-B0)
        # num_classes=0 removes the classifier, returns pooled features
        self.rgb_backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)
        self.rgb_dim = 1280 

        # Frequency Stream (FFT) - Input: (B, 1, H, W)
        self.fft_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fft_dim = 64

        # Error Stream (Noise Residual) - Input: (B, 3, H, W)
        self.noise_backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.noise_dim = 64

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(self.rgb_dim + self.fft_dim + self.noise_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1) # Logits
        )

    def forward(self, rgb, fft, noise):
        # RGB Stream
        rgb_feat = self.rgb_backbone(rgb) # (B, 1280)
        
        # FFT Stream
        fft_feat = self.fft_backbone(fft) # (B, 64)
        
        # Noise Stream
        noise_feat = self.noise_backbone(noise) # (B, 64)
        
        # Fuse
        combined = torch.cat([rgb_feat, fft_feat, noise_feat], dim=1)
        output = self.head(combined)
        return output
