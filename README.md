# Deepfake-Detection-Plugin-Chrome Extension
This repository houses the code for building your own deepfake detection plugin (chrome extension) developed by Bashaar Naik on 9th January, 2025!

# 🕵️ InnovateX: Deepfake Guard (Track 1)

**Deepfake Guard** is a real-time browser extension designed to detect AI-generated faces in video streams (YouTube, etc.) using a multi-model ensemble approach.

## 🚀 Key Features
- **Real-Time Analysis**: Captures frames directly from the active browser tab.
- **Ensemble AI Core**: Combines 4 distinct detection algorithms (CNN, ViT, CLIP, CV).
- **Transparent Breakdown**: Displays individual confidence scores for every model in a neat table.
- **Visual Dashboard**: Color-coded "Real" vs "Fake" badges with confidence percentages.

## 📂 Project Structure
Here is how the project files are organized:

```text
/DeepFake Detection Plugin
├── /deepfake-backend # 🗄️ Backend Server and Model Hosting

│   ├── server_master.py     
│   ├── server_ensemble.py         
│   ├── server_effnet.py
│   ├── server_openaiclip.py
│   ├── server_resnet.py
│   ├── server_vit.py
│   ├── server.py

├── requirements.txt       # 📦 Dependencies list for Python

├── /deepfake-extension    # 👁️ THE EYES: Chrome Extension Folder
│   ├── manifest.json      #    Config file (Permissions, Version)
│   ├── popup.html         #    The User Interface (Buttons, Table)
│   ├── popup.js           #    Logic (Talking to Python, Updating UI)
│   ├── content.js         #    Script injected into YouTube to capture video
│   └── icon.png           #    (Optional) Extension Icon
│
└── README.md              #    This documentation file

## 📦 **Installation**

### 1. Backend Setup (The Brain)
```bash
# Clone the repository
git clone [https://github.com/yourusername/innovatex-deepfake-guard.git](https://github.com/yourusername/innovatex-deepfake-guard.git)
cd innovatex-deepfake-guard

# Install dependencies
pip install -r requirements.txt

# Start the Server
python server_master.py
