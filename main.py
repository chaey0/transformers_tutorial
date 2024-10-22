import argparse
import random, os

import torch
import torch.nn.functional as F

import wandb
from torch import nn
from torch.utils.data import DataLoader

from train_CE import train, test
from models import *
from dataset import *
from DAT import *

from torchvision.models import regnet_y_16gf, efficientnet_b0, swin_b
from torchvision.models.vision_transformer import VisionTransformer as vision_transformer

seed = 42
# Python 기본 시드 설정
random.seed(seed)
# Numpy 시드 설정
np.random.seed(seed)
# Python 기본 시드 설정
random.seed(seed)
# Numpy 시드 설정
np.random.seed(seed)
# PyTorch 시드 설정
torch.manual_seed(seed)
# CUDA 사용 시 시드 설정
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
# CuDNN 일관성을 위해 다음 옵션 추가 (필요에 따라 선택)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(args):
    wandb.init(
        project="KBSMC",
        config={
            "learning_rate": args.init_lr,
            "architecture": args.model,
            "dataset": "KBSMC",
            "epochs": args.epochs,
        },
        name=f'{args.model}_{args.init_lr}'
    )

    transform = transforms.Compose([

        #transforms.CenterCrop((512, 512)),
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
    #test_dataset = DatasetSerial(test_set, img_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = (f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if args.model == 'EfficientNet':
        model = efficientnet_b0(pretrained=False)
        #model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # Adjust for 4 classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

        model = model.to(device)

    elif args.model == 'Swin':
        #model=SwinTransformer().to(device)

        model = swin_b()

        model.head = nn.Linear(model.head.in_features, 4)
        model.to(device)


    elif args.model == 'RegNet':
        model = regnet_y_16gf(pretrained=False).to(device)
        model.stem[0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(device)
        model.fc = nn.Linear(model.fc.in_features, 4).to(device)

    elif args.model== 'ViT':
        #model = ViT().to(device)
        model = vision_transformer(
            image_size=512,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3024,
            num_classes=4,
        ).to(device)

    elif args.model=='DAT_model':
        #model = DAT().to(device)
        model=DAT_model().to(device)

    elif args.model=='DAT':
        model = DAT().to(device)

    # 모델 출력 및 손실 계산

    #loss_fn = FocalLoss()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs // 3, T_mult=1, eta_min=args.init_lr* 0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

    print(f"start training with model: {args.model}")

    output_train_dir = f'KBSMC/Best_Model/train'
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)

    output_val_dir = f'KBSMC/Best_Model/val'
    if not os.path.exists(output_val_dir):
        os.makedirs(output_val_dir)

    best_train_loss = 100
    best_test_loss = 100

    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer, t + 1, device)
        test_loss, test_acc = test(valid_loader, model, loss_fn, t + 1, device)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model

            torch.save(best_model.state_dict(), f"{output_train_dir}/CE/{args.model.lower()}_{t + 1}_epoch_model.pth")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = model

            torch.save(best_model.state_dict(), f"{output_val_dir}/CE/{args.model.lower()}_{t + 1}_epoch_model.pth")

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        scheduler.step(test_loss)
        #scheduler.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script for FashionMNIST models")
    parser.add_argument('--cuda_num', type=int, default=5, help='CUDA device number to use')
    parser.add_argument('--init_lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--model', type=str, default='EfficientNet',
                        choices=['EfficientNet', 'ViT', 'Swin', 'RegNet', 'DAT','DAT_model'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

    args = parser.parse_args()

    main(args)

# python main.py --cuda_num 0 --model ResNet
# python main.py --cuda_num 1 --model EfficientNet
# python main.py --cuda_num 2 --model MobileNetV2
# python main.py --cuda_num 3 --model ViT
# python main.py --cuda_num 4 --model DAT --batch_size 16 --init_lr 5e-5

