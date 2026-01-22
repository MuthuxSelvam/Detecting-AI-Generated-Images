# Data Directory

This directory stores training and validation images for the AI Image Detector.

## Expected Structure

```
data/
├── train/
│   ├── real/     # Real images for training
│   └── fake/     # AI-generated images for training
├── val/
│   ├── real/     # Real images for validation
│   └── fake/     # AI-generated images for validation
└── test/         # Test images (optional)
```

## Quick Start (Dummy Data)

To generate dummy data for testing, run:

```bash
python verify_setup.py
```

## Getting Real Training Data

You can use datasets from:

1. **CIFAKE Dataset** - Real vs AI-generated images
   - [Kaggle CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

2. **Real Images**:
   - [COCO Dataset](https://cocodataset.org/)
   - [ImageNet](https://www.image-net.org/)

3. **AI-Generated Images**:
   - Generate using Stable Diffusion, MidJourney, DALL-E
   - [This Person Does Not Exist](https://thispersondoesnotexist.com/) for faces

## Image Requirements

- Supported formats: `.jpg`, `.jpeg`, `.png`
- Recommended size: 224x224 or larger (will be resized automatically)
- Minimum: 100 images per class for training, 20 for validation
