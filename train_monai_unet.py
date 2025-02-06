import torch
import numpy as np
import os
from Lacune_dataset import Lacune_dataset
from torchvision import transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, lr_scheduler
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
import torch.nn as nn
from eval import *
from monai.networks.nets import UNet
import natsort
from sklearn.model_selection import KFold, train_test_split

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

img_png_path = '/nasdata4/mjh/Diffusion/2_Segmentation/Local_Patch_Global_Attention/LPGA_lacune/data_2/fold'
gt_png_path = '/nasdata4/mjh/Diffusion/2_Segmentation/Local_Patch_Global_Attention/LPGA_lacune/data_2/fold_gt'

mri_png_paths = natsort.natsorted([os.path.join(img_png_path, f) for f in os.listdir(img_png_path) if f.endswith('.png')])
label_png_paths = natsort.natsorted([os.path.join(gt_png_path, f) for f in os.listdir(gt_png_path) if f.endswith('.png')])

transform = transforms.Compose([
    transforms.RandomCrop((224, 224)),  # 랜덤 크롭
    transforms.RandomRotation(degrees=10),  # 랜덤 회전
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomVerticalFlip(p=0.5),    # 상하 반전
    # RandomElasticDeformation(alpha=34, sigma=4),  # Elastic Deformation (사용자 정의 클래스)
    transforms.Resize((224, 224)),  # 스케일링 (사이즈 유지)
])

fold_idx = 1
# split = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=1234)

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
epochs = 500

for fold, (t_idx, v_idx) in enumerate(kf.split(mri_png_paths)):
# for t_idx, v_idx in split.split(mri_png_paths, label_png_paths):
    
    print(f"Fold {fold + 1}/{kf.n_splits} - Training: {len(t_idx)}, Validation: {len(v_idx)}")
    
    train_images = [mri_png_paths[i] for i in t_idx]
    train_labels = [label_png_paths[i] for i in t_idx]
    val_images = [mri_png_paths[i] for i in v_idx]
    val_labels = [label_png_paths[i] for i in v_idx]
    
    train_dataset = Lacune_dataset(train_images, train_labels, transform=transform)
    val_dataset = Lacune_dataset(val_images, val_labels)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    loaders = {"train": train_loader, "valid": val_loader}
    
    learning_rate = 0.0001
    
    weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/weights/weight_monai_unet"
    print("weights_path to save model parameters : ", weights_path)
    
    weights_best_name_tar = "Best_RLK-Unet_HF_Tar_" + str(fold_idx) + "_epoch500.pth"
    weights_best_tar = os.path.join(weights_path, weights_best_name_tar)
    
    weights_best_name_pxl = "Best_RLK-Unet_HF_Pxl_" + str(fold_idx) + "_epoch500.pth"
    weights_best_pxl = os.path.join(weights_path, weights_best_name_pxl)
    
    weights_best_name_avg = "Best_RLK-Unet_HF_Avg_" + str(fold_idx) + "_epoch500.pth"
    weights_best_avg = os.path.join(weights_path, weights_best_name_avg)
    
    # weights_best_name_tar_rec = "Best_RLK-Unet_HF_Tar_Rec_" + str(fold_idx) + "_epoch500.pth"
    # weights_best_tar_rec = os.path.join(weights_path, weights_best_name_tar_rec)
    
    # weights_best_name_pxl_rec = "Best_RLK-Unet_HF_Pxl_Rec_" + str(fold_idx) + "_epoch500.pth"
    # weights_best_pxl_rec = os.path.join(weights_path, weights_best_name_pxl_rec)
    
    # weights_best_name_avg_rec = "Best_RLK-Unet_HF_Avg_Rec" + str(fold_idx) + "_epoch500.pth"
    # weights_best_avg_rec = os.path.join(weights_path, weights_best_name_avg_rec)
    
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        dropout=0.1,
        num_res_units=2
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
    dice = DiceMetric(include_background=False, reduction='mean')
    criterion = DiceLoss(sigmoid=True, include_background="False")
    
    step = 0
    
    loss_train = []
    loss_valid = []
    all_loss_train = []
    all_dice_train = []
    
    all_val_metrics = []

    max_tar_f1 = 0
    max_pxl_f1 = 0
    max_avg_f1 = 0

    max_tar_rec = 0
    max_pxl_rec = 0
    max_avg_rec = 0
    
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
                y_data = torch.where(y_data > 0.0, 1.0, 0.0)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(x_data) # (B, 1, 224, 224)
                    total_loss = criterion(outputs, y_data)
                    
                    if phase == "train":
                        loss_train.append(total_loss.item())
                        y_pred = torch.where(outputs >= 0.5, 1.0, 0.0)
                        dice_score = dice(y_pred, y_data)
                        train_dice_list.append(dice_score.mean().item())
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
                        optimizer.step() # backpropagation
                    
                    if phase == "valid":
                        y_pred = torch.where(outputs >= 0.5, 1.0, 0.0) # (B, 1, 224, 224)
                        for k in range(y_pred.shape[0]):
                            y_pred_list.append(y_pred[k, 0, :, :])
                            y_gt_list.append(y_data[k, 0, :, :])
            
            if phase == "train":
                print("=================================================")
                print("epoch {}  | {}: {:.3f}".format(epoch + 1, "Train loss", np.mean(loss_train)))
                print("         | {}: {:.3f}".format("Train Dice", np.mean(train_dice_list)))
                all_loss_train.append(np.mean(loss_train))
                all_dice_train.append(np.mean(train_dice_list))
                loss_train  = []
        
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
        
        # if tarRec == 0 or pxlRec == 0:
        #     avg_rec = 0
        # else:
        #     avg_rec = 2/((1/tarRec) + (1/pxlRec))
        
        # scheduler.step(avg_f1)
        
        if tarf1 > max_tar_f1:
            max_tar_f1 = tarf1
            torch.save(model.state_dict(), weights_best_tar)
        print("Current Best Val target-level F1 Score : {:.3f}".format(max_tar_f1))
        
        # if tarRec > max_tar_rec:
        #     max_tar_rec = tarRec
        #     torch.save(model.state_dict(), weights_best_tar_rec)
        # print("Current Best Val target-level Recall : {:.3f}".format(max_tar_rec))
        
        if pxlf1 > max_pxl_f1:
            max_pxl_f1 = pxlf1
            torch.save(model.state_dict(), weights_best_pxl)
        print("Current Best Val pixel-level F1 Score : {:.3f}".format(max_pxl_f1))
        
        # if pxlRec > max_pxl_rec:
        #     max_pxl_rec = pxlRec
        #     torch.save(model.state_dict(), weights_best_pxl_rec)
        # print("Current Best Val pixel-level Recall : {:.3f}".format(max_pxl_rec))
        
        if avg_f1 > max_avg_f1:
            max_avg_f1 = avg_f1
            torch.save(model.state_dict(), weights_best_avg)
        print("Current Best Val average F1 Score : {:.3f}".format(max_avg_f1))
        
        # if avg_rec > max_avg_rec:
        #     max_avg_rec = avg_rec
        #     torch.save(model.state_dict(), weights_best_avg_rec)
        # print("Current Best Val average Recall : {:.3f}".format(max_avg_rec))
        
        all_val_metrics.append([tarf1, pxlf1, avg_f1])
    
    all_loss_train_np = np.array(all_loss_train)
    all_dice_train_np = np.array(all_dice_train)
    all_val_metrics_np = np.array(all_val_metrics)
    
    np.save(weights_path + '/' + 'train_loss_' + str(fold_idx) + '.npy', all_loss_train_np)
    np.save(weights_path + '/' + 'train_dice_' + str(fold_idx) + '.npy', all_dice_train_np)
    np.save(weights_path + '/' + 'val_metrics_' + str(fold_idx) + '.npy', all_val_metrics_np)
    
    fold_idx += 1
    
    print("Training Complete")
    print("")
    print("=================================================")

print("All training Finished")