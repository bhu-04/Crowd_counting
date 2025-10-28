# Crowd_counting
## CNN + Transformer Hybrid for Robust Crowd Counting

This project implements a **state-of-the-art crowd counting model** that fuses a **DenseNet-121 convolutional backbone** with **Multi-Scale Dilated Attention (MSDA)** and **Location-Enhanced Attention (LEA)** modules.  
Designed for datasets like **ShanghaiTech A**, this pipeline is flexible and can adapt to any crowd counting dataset in `.mat` format.

---

### 📁 Repository Structure
```
Crowd_counting/
├── data/
│   ├── train/
│   │   ├── images/     # Training images
│   │   └── annots/     # .mat annotation files
│   └── test/
│       ├── images/
│       └── annots/
├── outputs/            # Model checkpoints & prediction visualizations
├── src/
│   ├── model.py        # Model: DenseNet-121 + MSDA + LEA + regression head
│   ├── train.py        # Training script (supports resume)
│   ├── eval.py         # Evaluation & visualization
│   ├── dataset.py      # Custom dataset loader
│   └── utils.py        # Checkpointing, plotting, etc.
├── requirements.txt
└── README.md
```

---

### 🚀 Features
- ✅ DenseNet-121 backbone for strong local feature extraction  
- ✅ Transformer-inspired attention modules for global context  
- ✅ Robust regression head for precise crowd count prediction  
- ✅ Resume training seamlessly from the latest checkpoint  
- ✅ Visualization of predictions vs ground truth for every image  

---

### ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/bhu-04/Crowd_counting.git
cd Crowd_counting
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or
.env\Scriptsctivate    # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

### 📂 Dataset Access
The dataset used in this project is **not included** in the repository and is hosted securely on **Dropbox**.

To access the dataset:
1. Request access using this link: [Dropbox Dataset Link](https://www.dropbox.com/scl/fo/2lu0e2hyivphcr943as3m/AF44tvTT9URXmdxoZas_kJI?rlkey=9e9i5pdsas57cqwmweswspp9d&st=f6ml8wxc&dl=0)  
2. Once approved and downloaded, organize the dataset as follows:
   ```
   data/train/images/
   data/train/annots/
   data/test/images/
   data/test/annots/
   ```
3. Ensure annotation files are in `.mat` format for compatibility.  
If you don’t have access, you can also use publicly available datasets like **ShanghaiTech Part A**.

---

### 🧠 Usage

#### 🔹 Train the Model
```bash
python src/train.py
```
- Automatically resumes from `outputs/best_model.pth` if available.  
- Model checkpoints saved after each epoch.

#### 🔹 Evaluate the Model
```bash
python src/eval.py
```
- Generates predicted vs. ground-truth visualizations.  
- Saves metrics (MAE, MSE) and images in `outputs/`.

---

### 🧩 Model Details
| Component | Description |
|------------|-------------|
| **Backbone** | DenseNet-121 pretrained on ImageNet |
| **Attention** | Multi-Scale Dilated Attention (MSDA) + Location-Enhanced Attention (LEA) |
| **Regressor** | Fully-connected layers with dropout for robustness |

---

### 🧹 Data Preprocessing
All images are resized to **384×384** and normalized as per ImageNet standards:
```python
torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
```
Crowd counts are extracted automatically from `.mat` annotation files.

---

### 🧩 Troubleshooting
| Issue | Possible Cause |
|--------|----------------|
| Model predicts near 0 for all images | Missing normalization in `dataset.py` |
| Resume checkpoint fails | Use the latest valid `.pth` file in `outputs/` |
| Prediction errors too large | Verify `.mat` key extraction and annotation parsing |

---

### 📦 Requirements
```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.22
matplotlib >= 3.5
scikit-learn >= 1.1
pillow >= 9.0
tqdm >= 4.60
scipy >= 1.7
```

---

### 🤝 Contributing
Pull requests and issues are welcome!  
For questions or collaborations, contact **[bhu-04](https://github.com/bhu-04)**.

---
