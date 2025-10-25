# Detecting-AI-Generated-Images
Generative AI models like DALL·E, Midjourney, and StyleGAN2 can now create hyper-realistic synthetic images, posing major risks such as misinformation, identity fraud, copyright violations, and erosion of public trust. Without reliable automated detection, these images spread unchecked across digital platforms.
The project implements a Hybrid CNN-Vision Transformer (ViT) architecture for detecting AI
generated images with high accuracy, combining the strengths of convolutional neural networks for local feature extraction and vision transformers for global pattern recognition          
🏆 Presented at ICSEIT 2K25 Conference (April 11-12, 2025)**  
📖 Published: ISBN 978-93-342-0319

🚀 Features

Hybrid CNN-ViT Architecture: Integrates local texture analysis (CNN) with global structural assessment (Vision Transformers).

High Accuracy & Robustness: Achieves 92.6% correctness, with enhanced detection resilience against compression and noise (+28%).

Fast Processing: Capable of analyzing over 1,000 images per minute, suitable for real-time applications.

Explainability: Incorporates Grad-CAM visualizations to highlight manipulated regions within images.

Open-Source & API: Fully documented with deployment-ready REST API for integration into media verification workflows.

📁 Project Structure             
ai-image-detection/                   
├── README.md              
├── requirements.txt           
├── app.py                  
├── detect.py               
├── .gitignore          
├── LICENSE             
├── ai_image_detector.h           
├── templates/                                                                                   
│   └── index.html                                           
└── static/            
    └── uploads/                     
        └── .gitkeep     

    ├── visualizations/                   # Plots and heatmaps
    ├── logs/                             # Training and API logs
    └── reports/                          # Final evaluation reports

⚙️ Installation
1. Clone the repository
git clone https://github.com/MuthuxSelvam/Detecting-AI-Generated-Images.git
cd Detecting-AI-Generated-Images
2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
3. Install dependencies
pip install -r requirements.txt

▶️ Usage
1: Web Interface                  
python app.py

Open your browser:

http://127.0.0.1:5000/


Upload an image and get instant predictions with confidence scores.

Option 2: Command-Line Detection
Single Image
python detect.py --image path/to/image.jpg

Batch Detection
python detect.py --input_dir path/to/folder/


Output will show classification, confidence, and probabilities for each image.

📊 Model Performance
Model	       Accuracy	Precision	Recall	F1-Score
CNN Baseline    89.3%	 88.7%	     89.9%	 89.3%
ViT	            91.2%	 90.8%	     91.6%	 91.2%
Hybrid CNN-ViT	94.7%	 94.3%	     95.1%	 94.7%

🔍 Key Features

✅ Detect AI-generated images in real-time

✅ Pixel-level anomaly detection

✅ Metadata and compression artifact analysis

✅ Batch processing support

✅ Confidence scores with probability visualization

✅ Easy web and CLI integration

🛠 Technologies Used

Python 3.8+

TensorFlow / Keras (Deep Learning)

Flask (Web Application)

OpenCV (Image Processing)

NumPy (Numerical Computing)

HTML / CSS / JavaScript (Frontend UI)

🚀 Future Improvements

⚡ Deployment on Heroku / AWS / Render

🧠 Integration of Vision Transformers for higher accuracy

📱 Mobile app support (iOS / Android)

🌐 Multi-language detection and localization

📈 Real-time video frame detection



 
