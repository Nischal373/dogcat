# 🐶🐱 Dog vs Cat Image Classifier

A deep learning project that classifies images as **Dog** or **Cat** using transfer learning.  
Built with **PyTorch** and **TorchVision**, served with a **Gradio** web interface.

---

## 🚀 Features
- Transfer learning with [EfficientNet-B0] (pretrained on ImageNet)
- Data augmentation for robustness (flip, rotation, color jitter, etc.)
- Early stopping + learning rate scheduling
- Outputs **label + confidence score**
- Gradio app for interactive predictions
- TorchScript export for deployment

---
## 🛠️ Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Nischal373/dogcat.git
   cd dogcat

2.Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows

3.Install dependencies:
pip install -r requirements.txt

🔮 Inference
Gradio web app:
python predict.py

🙌 Acknowledgements

PyTorch<br>
TorchVision Models<br>
Gradio<br>
Dataset inspiration: Kaggle Dogs vs Cats<br>
