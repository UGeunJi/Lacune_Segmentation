from network import RLKunet, initialize_weight
import torch
import numpy as np
import os
from Lacune_dataset import Lacune_png_dataset
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW, lr_scheduler
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
import torch.nn as nn
from eval import *

seed = 1234
print("current seed : ", seed)
torch.manual_seed(seed)

device = torch.device('cuda:3')

'''
    Define Dataset
    - Train/Val : Extracted Lacune 2D axial slices from all subjects except test (sub-233) subject
    - test : Extracted all 2D axial slices (both with and without lacune slices) from test (sub-233) subject
    - all png files, image with 3-channel (flair, t1, t2) z-norm, padded to (224, 224), label image with 1-chanel padded to (224, 224)
    - transforms : Random Flips, Crop, Rotate, Resize (224, 224)
    
'''
train_img_png_path = '/nasdata4/mjh/Diffusion/2_Segmentation/Local_Patch_Global_Attention/LPGA_lacune/data_2/train'
val_img_png_path = '/nasdata4/mjh/Diffusion/2_Segmentation/Local_Patch_Global_Attention/LPGA_lacune/data_2/val'
test_img_png_path = '/nasdata4/mjh/Diffusion/2_Segmentation/Local_Patch_Global_Attention/LPGA_lacune/data_2/test'

transform = transforms.Compose([
    transforms.RandomCrop((224, 224)),  # 랜덤 크롭
    transforms.RandomRotation(degrees=10),  # 랜덤 회전
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomVerticalFlip(p=0.5),    # 상하 반전
    # RandomElasticDeformation(alpha=34, sigma=4),  # Elastic Deformation (사용자 정의 클래스)
    transforms.Resize((224, 224)),  # 스케일링 (사이즈 유지)
])

train_dataset = Lacune_png_dataset(img_path=train_img_png_path, transform=transform)
val_dataset = Lacune_png_dataset(img_path=val_img_png_path)
test_dataset = Lacune_png_dataset(img_path=test_img_png_path)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

loaders = {"train": train_loader, "valid": val_loader, "test": test_loader}

'''
    Model 설정
    - model : RLK-Unet
    - epochs : 500
    - lr : 0.0001, scheduler = ReduceLROnPlateau(max, factor=0.1, patience=10, cooldown=50)(avg_f1)
    - Train metric : Loss, Dice score
    - Valid metric : Target prec/recall/f1, Pixel prec/recall/f1, avg f1
'''

model = RLKunet(in_channels=3, out_channels=2, features=64, group_num=8).to(device)
model.apply(initialize_weight)

epochs = 500
learning_rate = 0.0001
fold_num = 5

weights_best_tar_list = []
weights_best_pxl_list = []
weights_best_avg_list = []
weights_best_tar_rec_list = []
weights_best_pxl_rec_list = []
weights_best_avg_rec_list = []

weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/weights/weight_rlk_base"
print("weights_path to save model parameters : ", weights_path)

for i in range(fold_num):
    weights_best_name_tar = "Best_RLK-Unet_HF_Tar_" + str(i+1) + "_epoch500.pth"
    weights_best_tar = os.path.join(weights_path, weights_best_name_tar)
    weights_best_tar_list.append(weights_best_tar)
    
    weights_best_name_pxl = "Best_RLK-Unet_HF_Pxl_" + str(i+1) + "_epoch500.pth"
    weights_best_pxl = os.path.join(weights_path, weights_best_name_pxl)
    weights_best_pxl_list.append(weights_best_pxl)
    
    weights_best_name_avg = "Best_RLK-Unet_HF_Avg_" + str(i+1) + "_epoch500.pth"
    weights_best_avg = os.path.join(weights_path, weights_best_name_avg)
    weights_best_avg_list.append(weights_best_avg)
    
    weights_best_name_tar_rec = "Best_RLK-Unet_HF_Tar_Rec_" + str(i+1) + "_epoch500.pth"
    weights_best_tar_rec = os.path.join(weights_path, weights_best_name_tar_rec)
    weights_best_tar_rec_list.append(weights_best_tar_rec)
    
    weights_best_name_pxl_rec = "Best_RLK-Unet_HF_Pxl_Rec_" + str(i+1) + "_epoch500.pth"
    weights_best_pxl_rec = os.path.join(weights_path, weights_best_name_pxl_rec)
    weights_best_pxl_rec_list.append(weights_best_pxl_rec)
    
    weights_best_name_avg_rec = "Best_RLK-Unet_HF_Avg_Rec" + str(i+1) + "_epoch500.pth"
    weights_best_avg_rec = os.path.join(weights_path, weights_best_name_avg_rec)
    weights_best_avg_rec_list.append(weights_best_avg_rec)

optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', cooldown=50)

dice = DiceMetric(include_background=False, reduction='mean')
criterion = DiceLoss(include_background=False, to_onehot_y=False, softmax=False, reduction="mean")

'''
    Begin training and validation
    - 5 fold Training
'''
for k in range(fold_num):
    
    step = 0
    
    loss_train = []
    loss_valid = []
    all_loss_train = []
    all_dice_train = []

    max_tar_f1 = 0
    max_pxl_f1 = 0
    max_avg_f1 = 0

    max_tar_rec = 0
    max_pxl_rec = 0
    max_avg_rec = 0