import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet_y_16gf
import wandb
from tqdm import tqdm

from models import DeiT
from dataset import *

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="KBSMC",
    config={
        "architecture": DeiT,
        "dataset": "KBSMC",
        "epochs": 60,
    },
    name=f'DeiT_{LEARNING_RATE}'
)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

train_set, valid_set, test_set = prepare_colon_tma_data()
train_dataset = DatasetSerial(train_set, img_transform=transform)
valid_dataset = DatasetSerial(valid_set, img_transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

teacher_model = regnet_y_16gf(weights=None).to(DEVICE)
teacher_model.stem[0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(DEVICE)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 1).to(DEVICE)

model_path = 'KBSMC/Best_Model/train/regnet_model.pth'
checkpoint = torch.load(model_path, map_location=DEVICE)
teacher_model.load_state_dict(checkpoint)
teacher_model.eval()

model = DeiT().to(DEVICE)

# MSELoss for single-class regression
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

# KL Divergence loss for distillation
distillation_criterion = nn.KLDivLoss(reduction='batchmean')

def train(student_model, teacher_model, train_loader, optimizer, criterion, distillation_criterion, device, tau=0.3):
    student_model.train()
    running_loss = 0.0
    total = 0

    tqdm_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch}")

    for data, target in tqdm_bar:
        data, target = data.to(device), target.to(device).view(-1, 1)  # Ensure target is reshaped for regression

        optimizer.zero_grad()

        # Forward pass for student model
        student_cls_output, student_distill_output = student_model(data)
        student_cls_output = student_cls_output.view(-1, 1)  # Ensure shape is compatible for regression
        student_distill_output = student_distill_output.view(-1, 1)

        with torch.no_grad():
            teacher_output = teacher_model(data)
            teacher_pred = teacher_output.view(-1, 1)  # Reshape to match student output

        # Compute losses
        cls_loss = criterion(student_cls_output, target.float())  # Regression loss
        distill_loss = criterion(student_distill_output, teacher_pred)  # Distillation loss

        loss = (cls_loss + distill_loss) * 0.5
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total += target.size(0)

    return running_loss / len(train_loader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0.0
    total, correct = 0, 0

    tqdm_bar = tqdm(test_loader, total=len(test_loader), desc=f"Testing Epoch {epoch}")

    with torch.no_grad():
        for data, target in tqdm_bar:
            data, target = data.to(device), target.to(device).view(-1, 1)  # Ensure target is reshaped for regression

            cls_output, distill_output = model(data)
            cls_output = cls_output.view(-1, 1)
            distill_output = distill_output.view(-1, 1)

            combined_output=(cls_output + distill_output)/2.0

            teacher_output = teacher_model(data)
            teacher_pred = teacher_output.view(-1, 1)  # Reshape to match student output

            cls_loss = criterion(cls_output, target.float())
            distill_loss = criterion(distill_output, teacher_pred)

            loss=(cls_loss + distill_loss) * 0.5

            test_loss += loss.item() * data.size(0)
            total += target.size(0)

            pred_rounded = torch.round(combined_output)
            correct += (pred_rounded == target).type(torch.float).sum().item()

    return test_loss / len(test_loader.dataset), correct/total

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, teacher_model, train_dataloader, optimizer, criterion, distillation_criterion, DEVICE)
        test_loss, test_acc = test(model, test_dataloader, criterion, DEVICE)

        scheduler.step(test_loss)

        print(f'Epoch {epoch}/{EPOCHS} - '
              f'Train Loss: {train_loss:.4f}'
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

    # Save the model
    torch.save(model.state_dict(), "KBSMC_DeiT.pth")
