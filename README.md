# 🕵️ AI Image Detector

A professional-grade AI image detector that distinguishes between **Real** and **AI-Generated** images using a Multi-Modal Ensemble approach.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

## ✨ Features

- **EfficientNet-B0** backbone for spatial feature extraction
- **FFT Analysis** for detecting periodic GAN/Diffusion patterns
- **Noise Residual Analysis** for statistical anomaly detection
- **FastAPI Backend** for high-performance inference
- **Streamlit Dashboard** for user-friendly interactions

## 🔧 Prerequisites

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Git** ([Download](https://git-scm.com/downloads))
- **GPU (Optional)**: NVIDIA GPU with CUDA for faster inference

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuthuxSelvam/Detecting-AI-Generated-Images.git
cd Detecting-AI-Generated-Images
```

### Step 2: Create Virtual Environment

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

</details>

<details>
<summary><b>🪟 Windows (CMD)</b></summary>

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

</details>

<details>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

</details>

### Step 3: Generate Sample Data (Optional)

```bash
python verify_setup.py
```

### Step 4: Run the Application

<details>
<summary><b>🪟 Windows (PowerShell)</b></summary>

```powershell
.\run_app.ps1
```

</details>

<details>
<summary><b>🪟 Windows (CMD)</b></summary>

```cmd
run_app.bat
```

</details>

<details>
<summary><b>🐧 Linux / 🍎 macOS</b></summary>

```bash
chmod +x run_app.sh
./run_app.sh
```

</details>

<details>
<summary><b>Manual Start (Any Platform)</b></summary>

**Terminal 1 - Start API:**
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - Start Dashboard:**
```bash
streamlit run dashboard.py
```

</details>

### Step 5: Open Dashboard

Navigate to: **http://localhost:8501**

---

## 📁 Project Structure

```
Detecting-AI-Generated-Images/
├── api/
│   └── main.py          # FastAPI backend
├── ml/
│   ├── analysis.py      # FFT & Noise analysis
│   ├── dataset.py       # Data loader with augmentations
│   ├── model.py         # EfficientNet-B0 Ensemble
│   └── train.py         # Training loop
├── data/                # Training data (see data/README.md)
├── dashboard.py         # Streamlit UI
├── best_model.pth       # Pre-trained model weights
├── verify_setup.py      # Generate dummy data
├── run_app.ps1          # Windows PowerShell script
├── run_app.bat          # Windows CMD script
├── run_app.sh           # Linux/macOS script
└── requirements.txt     # Dependencies
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Upload image for detection |

### Example API Usage

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
# {"filename": "image.jpg", "is_fake": true, "probability": 0.87, "label": "Fake"}
```

## 📊 Model Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RGB Image  │     │  FFT Map    │     │ Noise Map   │
│ (224x224x3) │     │ (224x224x1) │     │ (224x224x3) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│EfficientNet  │   │ CNN Encoder  │   │ CNN Encoder  │
│    B0        │   │   (64 dim)   │   │   (64 dim)   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  Fusion Head    │
                  │ (1280+64+64→1)  │
                  └────────┬────────┘
                           ▼
                      Real / Fake
```

## 🏋️ Training Your Own Model

```bash
# Prepare your data in data/train/real, data/train/fake, etc.
# Then run:
python -m ml.train --data_dir data --epochs 10 --batch_size 16
```

## 🔧 Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224x224 | Input image dimensions |
| Batch Size | 16 | Training batch size |
| Learning Rate | 1e-4 | Adam optimizer LR |
| Augmentations | JPEG, Blur, Flip | Data augmentation |

## ❓ Troubleshooting

<details>
<summary><b>Connection Error: "Target machine actively refused it"</b></summary>

The API server is not running. Start it with:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

</details>

<details>
<summary><b>CUDA out of memory</b></summary>

The model will automatically use CPU if CUDA is unavailable. To force CPU:
```python
# In api/main.py, change:
device = torch.device("cpu")
```

</details>

<details>
<summary><b>Module not found errors</b></summary>

Ensure your virtual environment is activated and dependencies are installed:
```bash
pip install -r requirements.txt
```

</details>

## 📝 License

This project is for educational purposes.

---

Created with ❤️ by [Muthu Selvam](https://github.com/MuthuxSelvam)
