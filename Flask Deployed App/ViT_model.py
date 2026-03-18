"""
Vision Transformer (ViT) Model for Plant Disease Detection
Uses Hugging Face's pretrained ViT-Base (google/vit-base-patch16-224)
Fine-tuned on the New Plant Diseases Dataset for 39-class classification.
"""

import torch
import torch.nn as nn
from torchvision import transforms

# Try to import timm (preferred), fallback to manual ViT
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import ViTForImageClassification, ViTFeatureExtractor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class PlantDiseaseViT(nn.Module):
    """
    Vision Transformer for Plant Disease Classification.
    Uses a pretrained ViT-Base model with a custom classification head
    for 39 plant disease classes.
    """

    def __init__(self, num_classes=39, pretrained=True, model_name='vit_base_patch16_224'):
        super(PlantDiseaseViT, self).__init__()

        if TIMM_AVAILABLE:
            # Use timm library for ViT (recommended)
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            # Replace the classification head with a custom one
            in_features = self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        elif HF_AVAILABLE:
            # Fallback: Use Hugging Face's ViT
            self.model = ViTForImageClassification.from_pretrained(
                'google/vit-base-patch16-224',
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            raise ImportError(
                "Either 'timm' or 'transformers' library is required. "
                "Install with: pip install timm  OR  pip install transformers"
            )

        self.using_timm = TIMM_AVAILABLE

    def forward(self, x):
        if self.using_timm:
            return self.model(x)
        else:
            outputs = self.model(x)
            return outputs.logits


def get_vit_transforms(img_size=224):
    """
    Get the image transforms for ViT inference.
    ViT expects images normalized with ImageNet mean/std.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


# Class mapping (same as CNN.py)
idx_to_classes = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Background_without_leaves',
    5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew',
    7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight',
    11: 'Corn___healthy',
    12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)',
    14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy',
    16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot',
    18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot',
    20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight',
    22: 'Potato___Late_blight',
    23: 'Potato___healthy',
    24: 'Raspberry___healthy',
    25: 'Soybean___healthy',
    26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch',
    28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot',
    30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight',
    32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot',
    36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus',
    38: 'Tomato___healthy'
}
