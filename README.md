# Detecting-AI-Generated-Images
Generative AI models like DALL·E, Midjourney, and StyleGAN2 can now create hyper-realistic synthetic images, posing major risks such as misinformation, identity fraud, copyright violations, and erosion of public trust. Without reliable automated detection, these images spread unchecked across digital platforms.
The project implements a Hybrid CNN-Vision Transformer (ViT) architecture for detecting AI
generated images with high accuracy, combining the strengths of convolutional neural networks for
 local feature extraction and vision transformers for global pattern recognition

Features:

Hybrid CNN-ViT Architecture: Integrates local texture analysis (CNN) with global structural assessment (Vision Transformers).

High Accuracy & Robustness: Achieves 92.6% correctness, with enhanced detection resilience against compression and noise (+28%).

Fast Processing: Capable of analyzing over 1,000 images per minute, suitable for real-time applications.

Explainability: Incorporates Grad-CAM visualizations to highlight manipulated regions within images.

Open-Source & API: Fully documented with deployment-ready REST API for integration into media verification workflows.
