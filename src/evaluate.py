import torch
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ==========================================
# 1. Settings
# ==========================================

DATA_DIR = "PlantVillage-Dataset/raw/color"
MODEL_PATH = "models/tomato_disease_mobilenetv3.pth"

CLASS_NAMES = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Healthy"
]


# ==========================================
# 2. Load Dataset
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

# Same split used during training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================================
# 3. Load Model
# ==========================================

model = mobilenet_v3_small(weights=None)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    4
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ==========================================
# 4. Predictions
# ==========================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ==========================================
# 5. Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES
    )
)


# ==========================================
# 6. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================\n")

print(cm)