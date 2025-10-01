# CrowdCCT: Crowd Counting via CNN and Transformer

## 📌 Overview
CrowdCCT is a deep learning project designed for **crowd counting** on the **ShanghaiTech Part A dataset**.  
The model uses a **hybrid architecture** of **Convolutional Neural Networks (CNN)** and **Transformer layers** to estimate crowd counts from images.

- **CNN (DenseNet-121 backbone)** extracts local visual features  
- **Transformer layers** capture global contextual information  
- **Combined features** improve crowd density estimation  

---

## 🚀 Features
- DenseNet-121 as backbone CNN  
- Custom Transformer layers for context understanding  
- L1 Loss (Mean Absolute Error) for counting accuracy  
- Configurable hyperparameters  
- Training, evaluation, and visualization of results  

---

## 📂 Project Structure
```
root/
├── venv/                     # Python Virtual Environment
├── data/                     # ShanghaiTech Part A Dataset
│   ├── test/images/          # Test images (IMG_XXX.jpg)
│   └── train/images/         # Training images (IMG_XXX.jpg)
├── src/                      # Source Code
│   ├── config.py             # Settings and dataset paths
│   ├── data_loader.py        # Dataset and transformations
│   ├── main.py               # Entry point (runs training)
│   ├── model.py              # CrowdCCT architecture
│   └── train_eval.py         # Training and evaluation logic
├── outputs/                  # Results and saved models
│   ├── crowdcct_best_model.pth   # Best model weights
│   └── training_history.png      # Training/MAE plots
├── requirements.txt          # Dependencies
└── README.md                 # Setup guide
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your_repo_url>
   cd image_analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the ShanghaiTech Part A dataset** and place files according to the structure above.

---

## 🔧 Configuration
Modify `src/config.py` to adjust:
- Device: `"cuda"` or `"cpu"`  
- Learning rate, weight decay, batch size  
- Number of Transformer layers and heads  
- Image resize and crop dimensions  
- Dataset paths  
- Output paths for saved models and plots  

---

## ▶️ Running the Project
From the `src` folder, run:
```bash
python src/main.py
```

This will:
- Load training and test datasets  
- Initialize the model  
- Train for the configured number of epochs  
- Evaluate on the test set after each epoch  
- Save the **best model**  
- Save training and evaluation plots  

---

## 📦 Dependencies
- `torch >= 2.0.0`  
- `torchvision >= 0.15.0`  
- `numpy >= 1.24.0`  
- `matplotlib >= 3.7.0`  
- `scikit-learn >= 1.3.0`  
- `Pillow >= 10.0.0`  

---

## 📊 Results
- Optimized for **Mean Absolute Error (MAE)** on the **ShanghaiTech Part A test set**  
- Training and test loss curves saved in:  
  ```
  outputs/training_history.png
  ```

---

## 🔧 Customization
- Modify Transformer architecture and training hyperparameters in `src/config.py`  
- Extend data augmentation in `src/data_loader.py`  
- Update training loop and evaluation metrics in `src/train_eval.py`  

---

## 📬 Contact
Please open **issues** or **pull requests** for questions, bug reports, or contributions.  
