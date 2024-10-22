import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import regnet_y_16gf
import wandb
from tqdm import tqdm
from dataset import *


from torchvision.transforms import ToTensor, Resize, Compose
from models import DeiT
from transformers import DeiTForImageClassificationWithTeacher, DeiTConfig

EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

wandb.init(
        project="KBSMC",
        config={
            "learning_rate": 5e-5,
            "architecture": DeiT,
            "dataset": "KBSMC",
            "epochs": 60,
        },
        name=f'DeiT_{LEARNING_RATE}_CE'
    )

transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1., 1., 1.])
    ])

train_set, valid_set, test_set = prepare_colon_tma_data()

train_dataset = DatasetSerial(train_set, img_transform=transform)
valid_dataset = DatasetSerial(valid_set, img_transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

teacher_model = regnet_y_16gf(pretrained=False).to(DEVICE)
teacher_model.stem[0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(DEVICE)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 4).to(DEVICE)

model_path = 'KBSMC/Best_Model/train/CE/regnet_56_epoch_model.pth'
checkpoint = torch.load(model_path, map_location=DEVICE)

teacher_model.load_state_dict(checkpoint)
teacher_model.eval()

model = DeiT().to(DEVICE)
'''
config = DeiTConfig()
config.num_labels=4
config.image_size=512

# Pretrained 사용 없이 새로 모델을 생성
model = DeiTForImageClassificationWithTeacher(config)
model.classifier = nn.Linear(model.config.hidden_size, 4)
model = model.to(DEVICE)
'''

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

# KL Divergence loss
distillation_criterion = nn.KLDivLoss(reduction='batchmean')

def label_smoothing(labels, num_classes, epsilon=0.1):
    smooth_labels = torch.full((labels.size(0), num_classes), epsilon / (num_classes - 1), device=labels.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - epsilon)

    return smooth_labels

def train(student_model, teacher_model, train_loader, optimizer, criterion, distillation_criterion, device, num_classes=4, epsilon=0.1, tau=0.3):
    student_model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    tqdm_bar = tqdm(train_loader, total=len(train_loader), desc=f"Training Epoch {epoch}")

    for data, target in tqdm_bar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        student_cls_output, student_distill_output = student_model(data)
        '''
        outputs = student_model(data)

        # Extract cls_logits and distillation_logits from the output
        student_cls_output = outputs.cls_logits
        student_distill_output = outputs.distillation_logits
        '''

        # Hard distillation with teacher model
        with torch.no_grad():
            teacher_output = teacher_model(data)
            teacher_pred = teacher_output.argmax(dim=1)  # Hard distillation

        smooth_teacher_labels = label_smoothing(teacher_pred, num_classes, epsilon=epsilon)

        cls_loss = criterion(student_cls_output, target)

        #distill_loss = distillation_criterion(F.log_softmax(student_distill_output/tau, dim=1), smooth_teacher_labels)
        distill_loss = criterion(student_distill_output, smooth_teacher_labels)

        loss = (cls_loss + distill_loss) *0.5
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        combined_output = (F.softmax(student_cls_output, dim=1) + F.softmax(student_distill_output, dim=1)) / 2.0
        _, predicted = combined_output.max(1)

        running_loss += loss.item() * data.size(0)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return running_loss / len(train_loader.dataset),  correct / total

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    tqdm_bar = tqdm(test_loader, total=len(test_loader), desc=f"Testing Epoch {epoch}")

    with torch.no_grad():
        for data, target in tqdm_bar:
            data, target = data.to(device), target.to(device)

            cls_output, distill_output = model(data)
            '''
            outputs = model(data)

            cls_output = outputs.cls_logits
            distill_output = outputs.distillation_logits
            '''
            teacher_output = teacher_model(data)
            teacher_pred = teacher_output.argmax(dim=1)  # Hard distillation

            smooth_teacher_labels = label_smoothing(teacher_pred, num_classes=4, epsilon=0.1)
            cls_loss = criterion(cls_output, target)
            distill_loss = criterion(distill_output, smooth_teacher_labels)

            loss = (cls_loss + distill_loss) * 0.5

            combined_output = (F.softmax(cls_output, dim=1) + F.softmax(distill_output, dim=1)) / 2.0
            _, predicted = combined_output.max(1)

            test_loss += loss.item() * data.size(0)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(test_loader.dataset),  correct / total

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, teacher_model, train_dataloader, optimizer, criterion, distillation_criterion, DEVICE)
        test_loss, test_acc = test(model, test_dataloader, criterion, DEVICE)

        scheduler.step(test_loss)

        print(f'Epoch {epoch}/{EPOCHS} - '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% - '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

    # 모델 저장
    torch.save(model.state_dict(), "KBSMC/Best_Model/train/CE/DeiT2.pth")

