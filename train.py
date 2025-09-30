import os
import torch
from src import data_setup, engine, model_builder, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.9
MAX_LR = 0.1
WEIGHT_DECAY = 1e-4

# Setup directories
train_dir = "../data"
test_dir = "../data"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
# Train Phase transformations
train_transforms = A.Compose([
    A.HorizontalFlip(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215827, 0.44653124), mask_fill_value=None),  # Apply coarse dropout
    A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),  # Normalize the image
    ToTensorV2() # Convert image to a PyTorch tensor
])


# Test Phase transformations
test_transforms = A.Compose([
    A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),  # Normalize the image
    ToTensorV2()  # Convert image to a PyTorch tensor
])


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.Model1().to(device)

# Set loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=NUM_EPOCHS, verbose=True)


# Start training with help from engine.py
engine.train(model=model,
             train_loader=train_dataloader,
             test_loader=test_dataloader,
             criterion=criterion,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             scheduler=scheduler)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="S9Model1.pth")
