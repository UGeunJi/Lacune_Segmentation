import os
import numpy as np
from PIL import Image
import natsort
import random
import natsort
import nibabel as nib
import torch
from Lacune_dataset import Lacune_fold_dataset, Lacune_fold_raw_dataset
from torch.utils.data import DataLoader
from network import RLKunet, initialize_weight
from torch.optim import AdamW
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
import torch.nn as nn
from eval import *
from utils import make_density
from torchvision import transforms as transforms
import matplotlib.pyplot as plt

def extract_lacune_slices(train_flair_files, train_t1_files, train_t2_files, train_label_files):
    
    slices_with_lacune = []

    for i in range(len(train_flair_files)):
        
        flair = nib.load(train_flair_files[i]).get_fdata()
        t1 = nib.load(train_t1_files[i]).get_fdata()
        t2 = nib.load(train_t2_files[i]).get_fdata()
        label = nib.load(train_label_files[i]).get_fdata()
        label = np.where(label > 0.0, 1.0, 0.0)
        
        for j in range(label.shape[2]):
            if np.max(label[:, :, j]) > 0.0:
                slices_with_lacune.append((flair[:, :, j], t1[:, :, j], t2[:, :, j], label[:, :, j]))
    
    return slices_with_lacune

seed = 1234
print("current seed : ", seed)
print("")
torch.manual_seed(seed)
random.seed(seed)

torch.set_num_threads(4)
device = torch.device('cuda:3')

lacune_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort_rlk_raw/with_lacune'

flair_path = os.path.join(lacune_path, "flair")
t1_path = os.path.join(lacune_path, "t1")
t2_path = os.path.join(lacune_path, "t2")
label_path = os.path.join(lacune_path, "Rater")

total_flair_list = natsort.natsorted(os.listdir(flair_path))
total_t1_list = natsort.natsorted(os.listdir(t1_path))
total_t2_list = natsort.natsorted(os.listdir(t2_path))
total_label_list = natsort.natsorted(os.listdir(label_path))

Folds_val_idx = [[2, 7, 9, 15],
                 [3, 8, 13, 21],
                 [0, 10, 11, 16],
                 [4, 5, 17, 18],
                 [1, 6, 12, 20]]

Folds_test_idx = [[4, 6, 13, 21],
             [0, 7, 15, 16],
             [3, 8, 9, 19],
             [1, 10, 14, 20],
             [2, 5, 11, 18]]

folds_num = 5
total_idx = list(range(22))

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomVerticalFlip(p=0.5)   # 상하 반전
    # RandomElasticDeformation(alpha=34, sigma=4),  # Elastic Deformation (사용자 정의 클래스)
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512))
])

weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/4_RLK_Ours/weights_raw/weight_density_multi_lambda_5fold_preset_raw_3000"

for fold in range(folds_num):
    
    print(f"Current Fold : {fold+1}")
    os.makedirs(weights_path + '/Fold_' + str(fold+1))
    
    test_fold_ith_index = Folds_test_idx[fold] # [2, 9, 16, 19]
    val_fold_ith_index = Folds_val_idx[fold] # [4, 8, 10]
    tmp_fold_ith_index = [k for k in total_idx if k not in test_fold_ith_index]
    train_fold_ith_index = [k for k in tmp_fold_ith_index if k not in val_fold_ith_index]
    
    train_fold_ith_flair_list = [flair_path + '/' + total_flair_list[index] for index in train_fold_ith_index]
    train_fold_ith_t1_list = [t1_path + '/' + total_t1_list[index] for index in train_fold_ith_index]
    train_fold_ith_t2_list = [t2_path + '/' + total_t2_list[index] for index in train_fold_ith_index]
    train_fold_ith_label_list = [label_path + '/' + total_label_list[index] for index in train_fold_ith_index]
    
    val_fold_ith_flair_list = [flair_path + '/' + total_flair_list[index] for index in val_fold_ith_index]
    val_fold_ith_t1_list = [t1_path + '/' + total_t1_list[index] for index in val_fold_ith_index]
    val_fold_ith_t2_list = [t2_path + '/' + total_t2_list[index] for index in val_fold_ith_index]
    val_fold_ith_label_list = [label_path + '/' + total_label_list[index] for index in val_fold_ith_index]
    
    train_lacune_slices = extract_lacune_slices(train_fold_ith_flair_list, train_fold_ith_t1_list, train_fold_ith_t2_list, train_fold_ith_label_list)
    val_lacune_slices = extract_lacune_slices(val_fold_ith_flair_list, val_fold_ith_t1_list, val_fold_ith_t2_list, val_fold_ith_label_list)
    
    train_dataset = Lacune_fold_raw_dataset(train_lacune_slices, transform=transform)
    val_dataset = Lacune_fold_raw_dataset(val_lacune_slices, transform=transform_val)
    
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    loaders = {"train": train_loader, "valid": val_loader}
    
    model = RLKunet(in_channels=3, out_channels=2, features=64, group_num=8).to(device)
    model.apply(initialize_weight)
    
    # epochs = 1000
    epochs = 3000
    learning_rate = 0.0001
    
    weights_fold_path = weights_path + '/Fold_' + str(fold+1)
    print("weights_path to save model parameters : ", weights_fold_path)
    
    np.save(weights_fold_path + '/' + "test_index.npy", test_fold_ith_index)
    np.save(weights_fold_path + '/' + "train_index.npy", train_fold_ith_index)
    np.save(weights_fold_path + '/' + "valid_index.npy", val_fold_ith_index)
    
    weights_best_name_tar    = "Best_RLK-Unet_HF_Tar_epoch600.pth"
    weights_best_tar         = os.path.join(weights_fold_path, weights_best_name_tar)

    weights_best_name_pxl    = "Best_RLK-Unet_HF_Pxl_epoch600.pth"
    weights_best_pxl        = os.path.join(weights_fold_path, weights_best_name_pxl)

    weights_best_name_avg    = "Best_RLK-Unet_HF_Avg_epoch600.pth"
    weights_best_avg       = os.path.join(weights_fold_path, weights_best_name_avg)

    weights_best_name_tar_rec    = "Best_RLK-Unet_HF_Tar_Rec_epoch600.pth"
    weights_best_tar_rec         = os.path.join(weights_fold_path, weights_best_name_tar_rec)

    weights_best_name_pxl_rec    = "Best_RLK-Unet_HF_Pxl_Rec_epoch600.pth"
    weights_best_pxl_rec        = os.path.join(weights_fold_path, weights_best_name_pxl_rec)

    weights_best_name_avg_rec    = "Best_RLK-Unet_HF_Avg_Rec_epoch600.pth"
    weights_best_avg_rec       = os.path.join(weights_fold_path, weights_best_name_avg_rec)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    dice = DiceMetric(include_background=False, reduction='mean')
    criterion = DiceLoss(include_background=False, to_onehot_y=False, softmax=False, reduction="mean")

    MSE = nn.MSELoss()
    
    step = 0
    loss_train = []
    loss_dice_train = []
    loss_density_train = []
    loss_valid = []
    all_loss_train = []
    all_loss_dice_train = []
    all_loss_density_train = []
    all_dice_train = []
    all_val_metrics = []

    max_tar_f1 = 0
    max_pxl_f1 = 0
    max_avg_f1 = 0

    max_tar_rec = 0
    max_pxl_rec = 0
    max_avg_rec = 0

    # initial_mse_weight = 0.1  # 초기 가중치
    initial_mse_weight = 1  # 초기 가중치
    final_mse_weight = 100
    
    ####### Starting Training #########
    print("Start training")
    
    for epoch in range(epochs):
        y_pred_list = []
        y_gt_list = []
        
        for phase in ["train", "valid"]:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            train_dice_list = []
            
            for idx, (x_data, y_data) in enumerate(loaders[phase]):
            
                if phase == "train":
                    step += 1
                
                x_data = x_data.to(device).float()
                y_data = y_data.to(device).float()
                # y_data = y_data.unsqueeze(1)
                y_data = torch.where(y_data > 0.0, 1.0, 0.0)
                
                maxpool = nn.MaxPool2d(2, 2)
                
                y_data_2 = maxpool(y_data)
                y_data_3 = maxpool(y_data_2)
                y_data_4 = maxpool(y_data_3)
                
                y_data_den = make_density(y_data, im_size=(512, 512), gaussian_std=2)
                y_data_den_2 = make_density(y_data_2, im_size=(256, 256), gaussian_std=3)
                y_data_den_3 = make_density(y_data_3, im_size=(128, 128), gaussian_std=5)
                y_data_den_4 = make_density(y_data_4, im_size=(64, 64), gaussian_std=7)
                
                y_data_den = torch.tensor(y_data_den, dtype=torch.float32).to(device) # (16, 1, 224, 224)
                y_data_den_2 = torch.tensor(y_data_den_2, dtype=torch.float32).to(device) # (16, 1, 112, 112)
                y_data_den_3 = torch.tensor(y_data_den_3, dtype=torch.float32).to(device) # (16, 1, 56, 56)
                y_data_den_4 = torch.tensor(y_data_den_4, dtype=torch.float32).to(device) # (16, 1, 28, 28)
                
                gt1 = torch.cat((y_data, 1 - y_data), dim=1)
                gt2 = torch.cat((y_data_2, 1 - y_data_2), dim=1)
                gt3 = torch.cat((y_data_3, 1 - y_data_3), dim=1)
                gt4 = torch.cat((y_data_4, 1 - y_data_4), dim=1)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred4, y_pred3, y_pred2, y_pred1 = model(x_data)
                    y_fg1, y_bg1, gt_fg1, gt_bg1 = y_pred1[:,0,:,:], y_pred1[:,1,:,:], gt1[:,0,:,:], gt1[:,1,:,:]
                    y_fg2, y_bg2, gt_fg2, gt_bg2 = y_pred2[:,0,:,:], y_pred2[:,1,:,:], gt2[:,0,:,:], gt2[:,1,:,:]
                    y_fg3, y_bg3, gt_fg3, gt_bg3 = y_pred3[:,0,:,:], y_pred3[:,1,:,:], gt3[:,0,:,:], gt3[:,1,:,:]
                    y_fg4, y_bg4, gt_fg4, gt_bg4 = y_pred4[:,0,:,:], y_pred4[:,1,:,:], gt4[:,0,:,:], gt4[:,1,:,:]
                    
                    y_fg1, y_bg1, gt_fg1, gt_bg1 = y_fg1.unsqueeze(1), y_bg1.unsqueeze(1), gt_fg1.unsqueeze(1), gt_bg1.unsqueeze(1)
                    y_fg2, y_bg2, gt_fg2, gt_bg2 = y_fg2.unsqueeze(1), y_bg2.unsqueeze(1), gt_fg2.unsqueeze(1), gt_bg2.unsqueeze(1)
                    y_fg3, y_bg3, gt_fg3, gt_bg3 = y_fg3.unsqueeze(1), y_bg3.unsqueeze(1), gt_fg3.unsqueeze(1), gt_bg3.unsqueeze(1)
                    y_fg4, y_bg4, gt_fg4, gt_bg4 = y_fg4.unsqueeze(1), y_bg4.unsqueeze(1), gt_fg4.unsqueeze(1), gt_bg4.unsqueeze(1)
                    
                    loss4 = criterion(y_fg4, gt_fg4) + criterion(y_bg4, gt_bg4)
                    loss3 = criterion(y_fg3, gt_fg3) + criterion(y_bg3, gt_bg3)
                    loss2 = criterion(y_fg2, gt_fg2) + criterion(y_bg2, gt_bg2)
                    loss1 = criterion(y_fg1, gt_fg1) + criterion(y_bg1, gt_bg1)
                    
                    loss = 0.1*(loss4) + 0.2*(loss3) + 0.3*(loss2) + 0.4*(loss1)
                    
                    loss1_den = MSE(y_fg1, y_data_den)
                    loss2_den = MSE(y_fg2, y_data_den_2)
                    loss3_den = MSE(y_fg3, y_data_den_3)
                    loss4_den = MSE(y_fg4, y_data_den_4)
                    
                    loss_den = 0.1*(loss4_den) + 0.2*(loss3_den) + 0.3*(loss2_den) + 0.4*(loss1_den)
                    
                    mse_weight = initial_mse_weight + (final_mse_weight - initial_mse_weight) * (epoch / epochs)
                    total_loss = loss + mse_weight * loss_den
                    
                    if phase == "train":
                        loss_train.append(total_loss.item())
                        loss_dice_train.append(loss.item())
                        loss_density_train.append(100 * loss_den.item())
                        y_pred = y_pred1[:,0,:,:]
                        y_pred = torch.unsqueeze(y_pred, 1)
                        y_pred = torch.where(y_pred >= 0.5, 1.0, 0.0)
                        dice_score = dice(y_pred, y_data)
                        train_dice_list.append(dice_score.mean().item())
                        total_loss.backward()
                        optimizer.step() # backpropagation
                    
                    if phase == "valid":
                        y_pred = y_pred1[:,0,:,:]
                        y_pred = torch.unsqueeze(y_pred, 1)
                        y_pred = torch.where(y_pred >= 0.5, 1.0, 0.0) # (B, 1, 224, 224)
                        
                        for k in range(y_pred.shape[0]):
                            y_pred_list.append(y_pred[k, 0, :, :])
                            y_gt_list.append(y_data[k, 0, :, :])
            
            if phase == "train":
                print("=================================================")
                print("epoch {}  | {}: {:.3f}, {}: {:.3f}".format(epoch + 1, "Train loss", np.nanmean(loss_train), "Train Dice Loss", np.nanmean(loss_dice_train)))
                print("         | {}: {:.3f}, {}: {:.3f}".format("Train Density Loss",np.nanmean(loss_density_train), "Train Dice Score", np.nanmean(train_dice_list)))
                all_loss_train.append(np.nanmean(loss_train))
                all_loss_dice_train.append(np.nanmean(loss_dice_train))
                all_loss_density_train.append(np.nanmean(loss_density_train))
                all_dice_train.append(np.nanmean(train_dice_list))
                loss_train  = []
                loss_dice_train  = []
                loss_density_train  = []
            
        
        print("..................................................")
        print("Validation : Target-level and Pixel-level Prec, Rec, F1")
        evaluator = Evaluator(y_pred_list, y_gt_list, tar_area=[0, np.inf], is_print=False)
        (Pd, Fa, TarPrec, TarRec, TarF1) = evaluator.target_metrics()
        (PxlPrec, PxlRec, PxlF1) = evaluator.pixel_metrics()
        
        tarPrec = TarPrec
        tarRec = TarRec
        tarf1 = TarF1
        pxlPrec = PxlPrec
        pxlRec = PxlRec
        pxlf1 = PxlF1
        
        if np.isnan(tarPrec) or np.isinf(tarPrec):
            tarPrec = 0
        if np.isnan(tarRec) or np.isinf(tarRec):
            tarRec = 0
        if np.isnan(tarf1) or np.isinf(tarf1):
            tarf1 = 0
        if np.isnan(pxlPrec) or np.isinf(pxlPrec):
            pxlPrec = 0
        if np.isnan(pxlRec) or np.isinf(pxlRec):
            pxlRec = 0
        if np.isnan(pxlf1) or np.isinf(pxlf1):
            pxlf1 = 0
        
        print("Current Val target-level Precision : {:.3f}, Recall : {:.3f}, and F1-score : {:.3f}".format(tarPrec, tarRec, tarf1))
        print("Current Val pixel-level Precision : {:.3f}, Recall : {:.3f}, and F1-score : {:.3f}".format(pxlPrec, pxlRec, pxlf1))
    
        if tarf1 == 0 or pxlf1 == 0:
            avg_f1 = 0
        else:
            avg_f1 = 2/((1/tarf1) + (1/pxlf1))
        
        if tarRec == 0 or pxlRec == 0:
            avg_rec = 0
        else:
            avg_rec = 2/((1/tarRec) + (1/pxlRec))
        
        if tarf1 > max_tar_f1:
            max_tar_f1 = tarf1
            torch.save(model.state_dict(), weights_best_tar)
        print("Current Best Val target-level F1 Score : {:.3f}".format(max_tar_f1))
        
        if tarRec > max_tar_rec:
            max_tar_rec = tarRec
            torch.save(model.state_dict(), weights_best_tar_rec)
        print("Current Best Val target-level Recall : {:.3f}".format(max_tar_rec))
        
        if pxlf1 > max_pxl_f1:
            max_pxl_f1 = pxlf1
            torch.save(model.state_dict(), weights_best_pxl)
        print("Current Best Val pixel-level F1 Score : {:.3f}".format(max_pxl_f1))
        
        if pxlRec > max_pxl_rec:
            max_pxl_rec = pxlRec
            torch.save(model.state_dict(), weights_best_pxl_rec)
        print("Current Best Val pixel-level Recall : {:.3f}".format(max_pxl_rec))
        
        if avg_f1 > max_avg_f1:
            max_avg_f1 = avg_f1
            torch.save(model.state_dict(), weights_best_avg)
        print("Current Best Val average F1 Score : {:.3f}".format(max_avg_f1))
        
        if avg_rec > max_avg_rec:
            max_avg_rec = avg_rec
            torch.save(model.state_dict(), weights_best_avg_rec)
        print("Current Best Val average Recall : {:.3f}".format(max_avg_rec))
        
        all_val_metrics.append([tarf1, pxlf1, avg_f1, tarRec, pxlRec, avg_rec])
    
    all_loss_train_np = np.array(all_loss_train)
    all_loss_dice_train_np = np.array(all_loss_dice_train)
    all_loss_density_train_np = np.array(all_loss_density_train)
    all_dice_train_np = np.array(all_dice_train)
    all_val_metrics_np = np.array(all_val_metrics)

    np.save(weights_fold_path + '/' + "train_loss.npy", all_loss_train_np)
    np.save(weights_fold_path + '/' + "train_dice_loss.npy", all_loss_dice_train_np)
    np.save(weights_fold_path + '/' + "train_density_loss.npy", all_loss_density_train_np)
    np.save(weights_fold_path + '/' + "train_dice.npy", all_dice_train_np)
    np.save(weights_fold_path + '/' + "val_metrics.npy", all_val_metrics_np)
    
    print("Training Complete")

print("")
print("All Folds are trained")