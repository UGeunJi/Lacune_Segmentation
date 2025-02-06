import os
import numpy as np
from PIL import Image
import natsort
import random
import natsort
import nibabel as nib
import torch
from Lacune_dataset import Lacune_fold_raw_dataset, Lacune_fold_dataset
from torch.utils.data import DataLoader
from network import RLKunet, initialize_weight
from torch.optim import AdamW
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
import torch.nn as nn
from eval import *
from torchvision import transforms as transforms

def extract_all_slices(train_flair_files, train_t1_files, train_t2_files, train_label_files):
    
    slices = []

    for i in range(len(train_flair_files)):
        
        flair = nib.load(train_flair_files[i]).get_fdata()
        t1 = nib.load(train_t1_files[i]).get_fdata()
        t2 = nib.load(train_t2_files[i]).get_fdata()
        label = nib.load(train_label_files[i]).get_fdata()
        label = np.where(label > 0.0, 1.0, 0.0)
        
        for j in range(label.shape[2]):
            slices.append((flair[:, :, j], t1[:, :, j], t2[:, :, j], label[:, :, j]))
    
    return slices

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

weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/4_RLK_Ours/weights_raw/weight_density_multi_lambda_5fold_preset_raw/Fold_5"
test_index = np.load(weights_path + '/' + "test_index.npy")

transform_test = transforms.Compose([
    transforms.Resize((512, 512))
])

test_flair_files = [flair_path + '/' + total_flair_list[index] for index in test_index]
test_t1_files = [t1_path + '/' + total_t1_list[index] for index in test_index]
test_t2_files = [t2_path + '/' + total_t2_list[index] for index in test_index]
test_label_files = [label_path + '/' + total_label_list[index] for index in test_index]

test_lacune_slices = extract_all_slices(test_flair_files, test_t1_files, test_t2_files, test_label_files)
test_dataset = Lacune_fold_raw_dataset(test_lacune_slices, transform=transform_test)

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = RLKunet(in_channels=3, out_channels=2, features=64, group_num=8).to(device)
weights_load_path = os.path.join(weights_path, "Best_RLK-Unet_HF_Tar_Rec_epoch600.pth")
model.load_state_dict(torch.load(weights_load_path, map_location='cuda:0'))
model.eval()

dice = DiceMetric(include_background=False, reduction='mean')
dice_score_list = []
y_truth_list = []
y_pred_list = []

x_data_plt_list = []
y_data_plt_list = []
y_pred_plt_list = []

# results_plt = np.zeros((192 * len(test_index), 224, 224*3))

selected_slices = []

with torch.no_grad():
    for idx, (x_data, y_data) in enumerate(test_loader):
        
        x_data = x_data.to(device).float()
        y_data = y_data.to(device).float()
        
        y_data = torch.where(y_data > 0.0, 1.0, 0.0)
        
        _, _, _, y_pred = model(x_data)
        y_pred_fg = y_pred[:, 0, :, :]
        y_pred_fg = y_pred_fg.unsqueeze(1)
        y_pred_fg = torch.where(y_pred_fg >= 0.5, 1.0, 0.0) # (1, 1, 224, 224)
        
        dice_score = dice(y_pred_fg, y_data)
        dice_score_list.append(dice_score.mean().item())
        
        y_pred_list.append(y_pred_fg[0, 0, :, :])
        y_truth_list.append(y_data[0, 0, :, :])
        
        # x_data_npy = x_data.detach().cpu().numpy()
        # results_plt[idx, :, :224] = x_data_npy[0, 2, :, :]
        # results_plt[idx, :, :224] = x_data_npy[0, 1, :, :]
        # x_data_plt_list.append(x_data_npy[0, :, :, :])
        
        # y_data_npy = y_data.detach().cpu().numpy()
        # results_plt[idx, :, 224:224*2] = y_data_npy[0, 0, :, :]
        # y_data_plt_list.append(y_data_npy[0, 0, :, :])
        
        # y_pred_fg_npy = y_pred_fg.detach().cpu().numpy()
        
        # if np.max(y_data_npy[0, 0, :, :]) > 0.0:
        #     selected_slices.append((x_data_npy[0, 2, :, :], y_data_npy[0, 0, :, :], y_pred_fg_npy[0, 0, :, :]))
        # results_plt[idx, :, 224*2:224*3] = y_pred_fg_npy[0, 0, :, :]
        # y_pred_plt_list.append(y_pred_fg_npy[0, 0, :, :])

evaluator = Evaluator(y_pred_list, y_truth_list, tar_area=[0, np.inf], is_print=False)

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

print("TarPrec : {:.3f}, TarRec : {:.3f}, and TarF1 : {:.3f}".format(tarPrec, tarRec, tarf1))
print("PxlPrec : {:.3f}, PxlRec : {:.3f}, and PxlF1 : {:.3f}".format(pxlPrec, pxlRec, pxlf1))

if tarf1 == 0 or pxlf1 == 0:
    avg_f1 = 0
else:
    avg_f1 = 2/((1/tarf1) + (1/pxlf1))
    
mean_test_dice_score = np.nanmean(dice_score_list)
    
print("Average F1 score : {:.3f}, mean dice score : {:.3f}".format(avg_f1, mean_test_dice_score))