import torch

# --- Project Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters (Based on Paper) ---
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EPOCHS = 10 
NUM_TRANSFORMER_LAYERS = 12
NUM_HEADS = 12

# --- Dataset Constants (ShanghaiTech Part-A) ---
IMAGE_RESIZE = (768, 1152) # Initial resize for input images
CROP_SIZE = 384            # Crop size for patch creation
NUM_CROPS = 6              # Number of patches per image (conceptual)

# --- Directory Paths (Adjusted to match your OS (C:)/image_analysis structure) ---
# Assuming you place the actual image/annot folders inside 'train' and 'test'
DATA_ROOT = './data'
TRAIN_IMG_DIR = f'{DATA_ROOT}/train/images'
TRAIN_ANNOT_DIR = f'{DATA_ROOT}/train/annots'
TEST_IMG_DIR = f'{DATA_ROOT}/test/images'
TEST_ANNOT_DIR = f'{DATA_ROOT}/test/annots'

# --- Model/Output ---
# Outputs will typically be saved in the project root or a designated 'outputs' folder.
MODEL_OUTPUT_PATH = './outputs/crowdcct_best_model.pth' 
HISTORY_PLOT_PATH = './outputs/training_history.png'