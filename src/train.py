import torch
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

print("PyTorch version:", torch.__version__)

# ==========================================
# 1. Dataset
# ==========================================

DATA_DIR = "PlantVillage-Dataset/raw/color"

weights = MobileNet_V3_Small_Weights.DEFAULT
transform = weights.transforms()

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# ==========================================
# 2. Train / Validation Split
# ==========================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==========================================
# 3. DataLoaders
# ==========================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 4. Model
# ==========================================

model = mobilenet_v3_small(weights=weights)

model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    4
)

for param in model.features.parameters():
    param.requires_grad = False


# ==========================================
# 5. Loss + Optimizer
# ==========================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# ==========================================
# 6. Device
# ==========================================

device = torch.device("cpu")

model = model.to(device)

print("Device:", device)


# ==========================================
# 7. Training
# ==========================================

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Training Loss: {average_loss:.4f}"
    )


print("Training completed!")


# ==========================================
# 8. Validation
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Validation Accuracy: {accuracy:.2f}%")


# ==========================================
# 9. Save Model
# ==========================================

MODEL_PATH = "models/tomato_disease_mobilenetv3.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    },
    MODEL_PATH
)

print("Model saved successfully!")
print("Saved to:", MODEL_PATH)