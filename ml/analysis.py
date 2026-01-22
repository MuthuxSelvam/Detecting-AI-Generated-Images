import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
import torch

def fft_analysis(image):
    """
    Performs FFT analysis on an image to detect periodic patterns (common in GANs).
    Args:
        image: numpy array (H, W, C) or (H, W) or torch tensor.
    Returns:
        magnitude_spectrum: Log-scaled magnitude spectrum.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Normalize to 0-1
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum) + 1e-8)
    
    return magnitude_spectrum

def noise_residual_analysis(image):
    """
    Extracts noise residuals using a denoising filter.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        
    # Simple denoising to get "smooth" image
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Residual = Original - Denoised
    residual = cv2.absdiff(image, denoised)
    
    # Normalize
    residual = residual / 255.0
    
    return residual
