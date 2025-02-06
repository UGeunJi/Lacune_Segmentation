import os
import numpy as np
import natsort
import torch
import random
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms as transforms
from tqdm import tqdm
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from utils.utils import *
from utils.eval import *
from Lacune_dataset import Lacune_fold_raw_dataset
from models.LPNetGA_v2 import LPNetGA as LPNetGA_v2

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
device = torch.device('cuda:0')

lacune_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort_rlk_raw/with_lacune_prep'

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

weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/3_LPNetGA/weights_raw/weight_LPNetGA_raw"

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
    
    model = LPNetGA_v2(im_size=(512, 512), ksize=(64, 64), stride=(32, 32)).to(device)
    
    epochs = 1000
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
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=200,
        cycle_mult=1.0,
        max_lr=1e-4,
        min_lr=1e-5,
        warmup_steps=100,
        gamma=0.8
    )
    
    MSE = nn.MSELoss()
    BCE = nn.BCELoss()
    
    step = 0
    loss_train = []
    loss_attn_train = []
    loss_lfe_train = []
    loss_valid = []
    all_loss_train = []
    all_loss_attn_train = []
    all_loss_lfe_train = []
    all_dice_train = []
    all_val_metrics = []

    max_tar_f1 = 0
    max_pxl_f1 = 0
    max_avg_f1 = 0

    max_tar_rec = 0
    max_pxl_rec = 0
    max_avg_rec = 0
    
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
                
                y_data_den = make_density(y_data, im_size=(512, 512), gaussian_std=5) # (B, 1, 512*512)
                y_data_subimgs = make_subimgs(y_data, im_size=(512, 512), ksize=(64, 64), stride=(32, 32)) # (B, patch_num, 64*64)
                
                y_data_den = torch.tensor(y_data_den, dtype=torch.float32).to(device)
                y_data_subimgs = torch.tensor(y_data_subimgs, dtype=torch.float32).to(device)
                
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    
                    (fusedFM, patches_tensor, attn_tensor) = model(x_data)
                    attn_loss = (model.im_size[0]*model.im_size[1]) * MSE(attn_tensor, y_data_den)
                    
                    lfe_loss = 0
                    for i, patch_tensor in enumerate(patches_tensor):
                        lfe_loss += BCE(patch_tensor, y_data_subimgs[:, i, :])/len(patches_tensor)
                    
                    loss = attn_loss + lfe_loss
                    
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss_attn_train.append(attn_loss.item())
                        loss_lfe_train.append(lfe_loss.item())
                        loss.backward()
                        optimizer.step()
                    
                    if phase == "valid":
                        fusedFM = fusedFM.reshape(y_data.shape)
                        fusedFM = fusedFM.detach().cpu().numpy()
                        
                        for k in range(fusedFM.shape[0]):
                            feature_map = fusedFM[k]
                            c, fh, fw = feature_map.shape
                            feature_map = np.expand_dims(feature_map, axis=0)
                            feature_map = bilinear_interpolate(feature_map, (512/fh, 512/fw), 0)
                            feature_map = np.reshape(feature_map, (1, 1, 512, 512))
                            feature_map = np.where(feature_map >= 0.5, 1.0, 0.0)
                            y_pred_list.append(feature_map[0, 0, :, :])
                            y_gt_list.append(y_data[k, 0, :, :])
            
            if phase == "train":
                print("=================================================")
                print("epoch {}  | {}: {:.3f}, {}: {:.3f}".format(epoch + 1, "Train loss", np.nanmean(loss_train), "Train Attention Loss", np.nanmean(loss_attn_train)))
                print("         | {}: {:.3f}".format("Train LFE Loss",np.nanmean(loss_lfe_train)))
                all_loss_train.append(np.nanmean(loss_train))
                all_loss_attn_train.append(np.nanmean(loss_attn_train))
                all_loss_lfe_train.append(np.nanmean(loss_lfe_train))
                loss_train  = []
                loss_attn_train  = []
                loss_lfe_train  = []
        
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
        
        scheduler.step()
    
    all_loss_train_np = np.array(all_loss_train)
    all_loss_attn_train_np = np.array(all_loss_attn_train)
    all_loss_lfe_train_np = np.array(all_loss_lfe_train)
    # all_dice_train_np = np.array(all_dice_train)
    all_val_metrics_np = np.array(all_val_metrics)

    np.save(weights_fold_path + '/' + "train_loss.npy", all_loss_train_np)
    np.save(weights_fold_path + '/' + "train_attn_loss.npy", all_loss_attn_train_np)
    np.save(weights_fold_path + '/' + "train_lfe_loss.npy", all_loss_lfe_train_np)
    # np.save(weights_fold_path + '/' + "train_dice.npy", all_dice_train_np)
    np.save(weights_fold_path + '/' + "val_metrics.npy", all_val_metrics_np)
    
    print("Training Complete")

print("")
print("All Folds are trained")