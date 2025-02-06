import torch
import random
import torch.nn as nn
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from utils.eval import *
from Lacune_dataset import Lacune_fold_raw_dataset, Lacune_png_all_modal_dataset
from models.LPNetGA_v2 import LPNetGA as LPNetGA_v2

def save_overlay_image(x_data, y_data, y_pred, idx, weights_path):
    
    save_path = weights_path + '/' + 'figure'
    
    x_data_np = x_data[0, 0].cpu().numpy()
    x_data_img = Image.fromarray((x_data_np * 255).astype(np.uint8)).convert('L')
    x_data_img = x_data_img.convert('RGBA')
    y_data_np = y_data[0, 0].cpu().numpy()
    y_pred_np = y_pred[0, 0].cpu().numpy()
    overlay_img = Image.new('RGBA', x_data_img.size, (0, 0, 0, 0))
    overlay_data = []
    for yd, yp in zip(y_data_np.flatten(), y_pred_np.flatten()):
        if yd == 1 and yp == 1:  # 겹치는 부분은 파란색
            overlay_data.append((0, 0, 255, 255))
        elif yd == 1:  # y_data는 빨간색
            overlay_data.append((255, 0, 0, 255))
        elif yp == 1:  # y_pred는 노란색
            overlay_data.append((255, 255, 0, 255))
        else:  # 배경은 투명
            overlay_data.append((0, 0, 0, 0))
    overlay_img.putdata(overlay_data)
    final_img = Image.alpha_composite(x_data_img, overlay_img)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(x_data_np, cmap='gray')
    ax[0].set_title('Original Image (Grayscale)')
    ax[0].axis('off')
    ax[1].imshow(final_img)
    ax[1].set_title('Overlay (Red: GT, Yellow: Pred, Blue: Overlap)')
    ax[1].axis('off')
    plt.savefig(os.path.join(save_path, 'overlay_' + str(idx) + '.png'), bbox_inches='tight')
    plt.close(fig)

seed = 1234
print("current seed : ", seed)
print("")
torch.manual_seed(seed)
random.seed(seed)

torch.set_num_threads(4)
device = torch.device('cuda:2')

img_path = '/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/4_RLK_Ours/Experiments/dataset'

fold = 1
img_fold_path = img_path + '/' + 'Fold_' + str(fold + 1)
test_path = img_fold_path + '/' + 'test'

transform_test = transforms.Compose([
    transforms.Resize((512, 512))
])
test_dataset = Lacune_png_all_modal_dataset(img_path=test_path, transform=transform_test)

batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

weights_path    = "/nasdata4/mjh/Diffusion/2_Segmentation/1_RLK_final/3_LPNetGA/weights_raw/weight_LPNetGA_raw/Fold_2"
model = LPNetGA_v2(im_size=(512, 512), ksize=(64, 64), stride=(32, 32)).to(device)
weights_load_path = os.path.join(weights_path, "Best_RLK-Unet_HF_Pxl_epoch600.pth")
model.load_state_dict(torch.load(weights_load_path, map_location="cuda:2"))
model.eval()

y_truth_list = []
y_pred_list = []

x_data_plt_list = []
y_data_plt_list = []
y_pred_plt_list = []

with torch.no_grad():
    for idx, (x_data, y_data) in enumerate(test_loader):
        
        x_data = x_data.to(device).float()
        y_data = y_data.to(device).float()
        
        y_data = torch.where(y_data > 0.0, 1.0, 0.0)
        
        (fusedFM, _, _) = model(x_data, mode='test')
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
            y_truth_list.append(y_data[k, 0, :, :])

# evaluator = Evaluator(y_pred_list, y_truth_list, tar_area=[0, np.inf], is_print=False)
# (Pd, Fa, TarPrec, TarRec, TarF1) = evaluator.target_metrics()
# (PxlPrec, PxlRec, PxlF1) = evaluator.pixel_metrics()

# tarPrec = TarPrec
# tarRec = TarRec
# tarf1 = TarF1
# pxlPrec = PxlPrec
# pxlRec = PxlRec
# pxlf1 = PxlF1

# if np.isnan(tarPrec) or np.isinf(tarPrec):
#     tarPrec = 0
# if np.isnan(tarRec) or np.isinf(tarRec):
#     tarRec = 0
# if np.isnan(tarf1) or np.isinf(tarf1):
#     tarf1 = 0
# if np.isnan(pxlPrec) or np.isinf(pxlPrec):
#     pxlPrec = 0
# if np.isnan(pxlRec) or np.isinf(pxlRec):
#     pxlRec = 0
# if np.isnan(pxlf1) or np.isinf(pxlf1):
#     pxlf1 = 0

# print("TarPrec : {:.3f}, TarRec : {:.3f}, and TarF1 : {:.3f}".format(tarPrec, tarRec, tarf1))
# print("PxlPrec : {:.3f}, PxlRec : {:.3f}, and PxlF1 : {:.3f}".format(pxlPrec, pxlRec, pxlf1))

# if tarf1 == 0 or pxlf1 == 0:
#     avg_f1 = 0
# else:
#     avg_f1 = 2/((1/tarf1) + (1/pxlf1))
    
# mean_test_dice_score = evaluator.dice_metrics()
# print("Average F1 score : {:.3f}, mean dice score : {:.3f}".format(avg_f1, mean_test_dice_score))
# a = 1