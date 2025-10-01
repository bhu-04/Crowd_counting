import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import config

class TransformerLayer(nn.Module):
    """Simplified Transformer Layer representing the combined MSDA/LEA logic."""
    def __init__(self, dim, num_heads):
        super().__init__()
        # Norm1 and Attention represent the core feature mixing (Global/Scale Context)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # Norm2 and MLP (Multi-Layer Perceptron) for feature transformation
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Attention with Residual Connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with Residual Connection
        x = x + self.mlp(self.norm2(x))
        return x

class CrowdCCT(nn.Module):
    """The Crowd Counting via CNN and Transformer (CrowdCCT) Model."""
    def __init__(self, num_transformer_layers=config.NUM_TRANSFORMER_LAYERS, 
                 num_heads=config.NUM_HEADS):
        super(CrowdCCT, self).__init__()

        # 1. CNN Part: DenseNet121 Feature Extractor (Local Details)
        densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        
        # Use the features part and remove the final classification layers
        self.cnn_features = nn.Sequential(*list(densenet.children())[0])
        
        # Channel Reduction (1024 -> 768)
        self.feature_reduction = nn.Conv2d(1024, 768, kernel_size=1) 
        
        # 2. Transformer Part (Global Context)
        self.transformer_block = nn.Sequential(
            *[TransformerLayer(768, num_heads) for _ in range(num_transformer_layers)]
        )
        
        # 3. Regression Head Part (Count Prediction)
        # Approximate the input size after feature extraction and flattening: 
        # (384x384 input -> 32x32 to 12x12 feature map size is common)
        # We assume a fixed size here for the model to compile.
        INPUT_FLATTENED_SIZE = 768 * 12 * 12 
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=INPUT_FLATTENED_SIZE, out_features=1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.8), 
            nn.Linear(512, 1)  # Output a single count value
        )

    def forward(self, x):
        # 1. CNN Pass
        x = self.cnn_features(x)
        x = self.feature_reduction(x)
        print("Feature map shape after reduction:", x.shape)

        # 2. Prepare for Transformer (C, H, W -> H*W, C)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. Transformer Pass
        x = self.transformer_block(x)
        
        # 4. Prepare for Regression (Flatten)
        x = x.transpose(1, 2).reshape(B, -1)
        
        # Handle dynamic sizing for the first FC layer only if needed (not shown here)
        
        count = self.regressor(x)
        return count.squeeze(-1)