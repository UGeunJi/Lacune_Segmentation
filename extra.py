import torch
import numpy as np
import os
import nibabel as nib
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from sklearn.preprocessing import QuantileTransformer
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 4D 이미지 데이터 (C, X, Y, Z)
# C: 모달리티 개수 (예: FLAIR, T1w, T2w 등)
# X, Y, Z: 공간 좌표
def create_nonzero_mask(image_data_4d):
    
    num_modalities = image_data_4d.shape[0]
    nonzero_mask = np.zeros(image_data_4d.shape[1:], dtype=bool)
    
    for modality_idx in range(num_modalities):
        modality_data = image_data_4d[modality_idx, :, :, :]  # 모달리티별 3D 데이터
        nonzero_mask = np.logical_or(nonzero_mask, modality_data != 0)  # non-zero 영역 결합
    
    filled_nonzero_mask = binary_fill_holes(nonzero_mask)
    
    return filled_nonzero_mask

def find_bounding_box(nonzero_mask):
    
    # 각 축(X, Y, Z)에서 1이 있는 최소 및 최대 좌표를 계산
    nonzero_indices = np.where(nonzero_mask)

    min_x, max_x = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_y, max_y = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    min_z, max_z = np.min(nonzero_indices[2]), np.max(nonzero_indices[2])

    return min_x, max_x, min_y, max_y, min_z, max_z

def crop_image_according_to_bbox(image_data_4d, bbox):
    
    min_x, max_x, min_y, max_y, min_z, max_z = bbox
    
    # 각 모달리티를 bounding box에 맞게 자르고 다시 결합
    cropped_modalities = []
    for modality_idx in range(image_data_4d.shape[0]):
        modality_data = image_data_4d[modality_idx, :, :, :]  # 각 모달리티의 3D 데이터
        cropped_modality = modality_data[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]  # 자르기
        cropped_modalities.append(cropped_modality)
    
    # 다시 4차원 배열로 결합 (C축으로 스택)
    cropped_image_data_4d = np.stack(cropped_modalities, axis=0)
    
    return cropped_image_data_4d

def calculate_target_spacing(spacing_list):
    
    spacing_array = np.array(spacing_list)
    target_spacing = np.median(spacing_array, axis=0)  # 각 축의 중간값 계산
    return tuple(target_spacing)

def resample_image(image_data, original_spacing, target_spacing):
    
    original_shape = np.array(image_data.shape)
    original_physical_size = original_spacing * original_shape
    
    # 목표 이미지의 shape는 목표 spacing에 맞춰 계산됨
    target_shape = np.round(original_physical_size / target_spacing).astype(int)
    
    # skimage의 resize 함수를 사용하여 이미지 리사이징
    resized_image = resize(image_data, target_shape, mode='reflect', anti_aliasing=True, preserve_range=True)
    
    return resized_image

def resample_all_images(image_data_4d, spacing_list, target_spacing):
    
    resized_modalities = []
    for modality_idx in range(image_data_4d.shape[0]):
        modality_data = image_data_4d[modality_idx, :, :, :]  # 각 모달리티의 3D 데이터
        original_spacing = spacing_list[modality_idx]  # 해당 모달리티의 원본 spacing
        resized_modality = resample_image(modality_data, original_spacing, target_spacing)
        resized_modalities.append(resized_modality)
    
    # 다시 4차원 배열로 결합 (C축으로 스택)
    resized_image_data_4d = np.stack(resized_modalities, axis=0)
    
    return resized_image_data_4d
    

# flair_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/flair_mni/sub-102_space-T1_desc-masked_linreg_FLAIR.nii.gz'
# t1_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t1_mni/sub-102_space-T1_desc-masked_linreg_T1.nii.gz'
# t2_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t2_mni/sub-102_space-T1_desc-masked_linreg_T2.nii.gz'
# Rater_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/Rater_mni/sub-102_space-T1_desc-Rater_linreg_Lacunes.nii.gz'

# flair_path = '/nasdata4/mjh/VALDO/tar/LACUNE/data/sub-233_space-T1_desc-masked_FLAIR.nii.gz'
# t1_path = '/nasdata4/mjh/VALDO/tar/LACUNE/data_t1/sub-233_space-T1_desc-masked_T1.nii.gz'
# t2_path = '/nasdata4/mjh/VALDO/tar/LACUNE/data_t2/sub-233_space-T1_desc-masked_T2.nii.gz'
# Rater_path = '/nasdata4/mjh/VALDO/tar/LACUNE/Rater/nifti/sub-233_space-T1_desc-Rater_Lacunes.nii.gz'

# flair = nib.load(flair_path).get_fdata()
# t1 = nib.load(t1_path).get_fdata()
# t2 = nib.load(t2_path).get_fdata()
# label = nib.load(Rater_path).get_fdata()

# image_4d = np.stack([flair, t1, t2], axis = 0)

# nonzero_mask = create_nonzero_mask(image_4d)
# bbox = find_bounding_box(nonzero_mask)
# print("Bounding box : ", bbox)

# cropped_image_4d = crop_image_according_to_bbox(image_4d, bbox)
# print("Cropped image shape:", cropped_image_4d.shape)

# a = 1
# spacing_list_1 = [(1.09, 1.09, 1.0), (1.09, 1.09, 1.0), (1.09, 1.09, 1.0)]
# spacing_list_2 = [(0.49, 0.49, 0.8), (0.49, 0.49, 0.8), (0.49, 0.49, 0.8)]

# target_spacing = calculate_target_spacing(spacing_list_2)
# print("Target spacing:", target_spacing)

# resized_image_data_4d = resample_all_images(cropped_image_4d, spacing_list_2, target_spacing)

# print("Resized image shape:", resized_image_data_4d.shape)

# b = 1

# def quantile_normalization(data):
    
#     data_flat = data.contiguous().view(data.shape[0], -1).cpu().numpy()
    
#     transformer = QuantileTransformer(output_distribution='normal', random_state=0)
#     data_normalized = transformer.fit_transform(data_flat)
    
#     # 정규화된 데이터를 다시 원래 형태로 변환 (N, C, H, W)
#     data_normalized = torch.tensor(data_normalized).view(data.shape).to(data.device)
    
#     return data_normalized

# flair_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/flair_mni/sub-102_space-T1_desc-masked_linreg_FLAIR.nii.gz'
# t1_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t1_mni/sub-102_space-T1_desc-masked_linreg_T1.nii.gz'
# t2_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t2_mni/sub-102_space-T1_desc-masked_linreg_T2.nii.gz'
# Rater_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/Rater_mni/sub-102_space-T1_desc-Rater_linreg_Lacunes.nii.gz'

# flair = nib.load(flair_path).get_fdata()
# t1 = nib.load(t1_path).get_fdata()
# t2 = nib.load(t2_path).get_fdata()
# label = nib.load(Rater_path).get_fdata()

# image_4d = np.stack([flair, t1, t2], axis = 0)
# image_4d_tensor = torch.tensor(image_4d).float()
# image_4d_tensor = torch.unsqueeze(image_4d_tensor, dim=0)

# normalized_data = quantile_normalization(image_4d_tensor)
# a = 1

def sample_random_crop(image, label, crop_size, foreground_prob=0.95):
    """
    192x192x32 크기의 크롭을 랜덤하게 샘플링합니다.
    95% 확률로 전경 중심에서 크롭을 선택하고, 그렇지 않으면 배경에서 선택합니다.
    
    Args:
        image (torch.Tensor): 입력 이미지 데이터 (C, H, W, D).
        label (torch.Tensor): 레이블 데이터 (H, W, D).
        crop_size (tuple): 크롭 크기 (H_crop, W_crop, D_crop).
        foreground_prob (float): 전경 중심에서 크롭할 확률.
    
    Returns:
        torch.Tensor: 크롭된 이미지 데이터.
        torch.Tensor: 크롭된 레이블 데이터.
    """
    # 전경(lesion)이 포함된 위치 찾기
    foreground_voxels = (label > 0).nonzero(as_tuple=False)
    
    # 랜덤하게 크롭의 중심을 선택
    if np.random.rand() < foreground_prob and len(foreground_voxels) > 0:
        # 전경에서 크롭 중심 선택
        center = foreground_voxels[np.random.randint(len(foreground_voxels))]
    else:
        # 배경에서 크롭 중심 선택
        center = torch.tensor([np.random.randint(s) for s in label.shape])
    
    # 크롭 시작 및 끝 좌표 계산
    start = [max(0, center[i] - crop_size[i] // 2) for i in range(3)]
    end = [min(image.shape[i + 1], start[i] + crop_size[i]) for i in range(3)]
    
    # 크롭된 이미지와 레이블 반환
    cropped_image = image[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    cropped_label = label[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # 크롭된 이미지와 레이블이 원하는 크기인지 확인하고 패딩
    cropped_image = F.pad(cropped_image, pad=(0, crop_size[2] - cropped_image.shape[3],
                                              0, crop_size[1] - cropped_image.shape[2],
                                              0, crop_size[0] - cropped_image.shape[1]), mode='constant', value=0)
    
    cropped_label = F.pad(cropped_label, pad=(0, crop_size[2] - cropped_label.shape[2],
                                              0, crop_size[1] - cropped_label.shape[1],
                                              0, crop_size[0] - cropped_label.shape[0]), mode='constant', value=0)
    
    return cropped_image, cropped_label

def create_batch(images, labels, batch_size=4, crop_size=(192, 192, 32)):
    """
    배치당 4개의 크롭을 생성합니다. 각 배치 요소는 랜덤하게 샘플링된 192x192x32 크기의 크롭을 포함합니다.
    
    Args:
        images (torch.Tensor): 입력 이미지 데이터 (N, C, H, W, D).
        labels (torch.Tensor): 레이블 데이터 (N, H, W, D).
        batch_size (int): 배치 크기.
        crop_size (tuple): 크롭 크기 (H_crop, W_crop, D_crop).
    
    Returns:
        torch.Tensor: 배치 크기 (batch_size, C, H_crop, W_crop, D_crop)의 크롭된 이미지 데이터.
        torch.Tensor: 배치 크기 (batch_size, H_crop, W_crop, D_crop)의 크롭된 레이블 데이터.
    """
    cropped_images = []
    cropped_labels = []
    
    for i in range(batch_size):
        # 배치 요소당 192x192x32 크기의 크롭 샘플링
        idx = np.random.randint(images.shape[0])  # N개의 이미지 중에서 랜덤 선택
        for _ in range(2):
            cropped_image, cropped_label = sample_random_crop(images[idx], labels[idx], crop_size)
            cropped_images.append(cropped_image)
            cropped_labels.append(cropped_label)
    
    # 배치로 결합
    batch_images = torch.stack(cropped_images)
    batch_labels = torch.stack(cropped_labels)
    
    return batch_images, batch_labels

# 사용 예시
# (N, C, H, W, D) 크기의 임의의 3D 이미지 데이터 생성 (예: 8개의 이미지, 3채널, 256x256x64 크기)

torch.manual_seed(1234)

flair_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/flair_mni/sub-102_space-T1_desc-masked_linreg_FLAIR.nii.gz'
t1_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t1_mni/sub-102_space-T1_desc-masked_linreg_T1.nii.gz'
t2_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/t2_mni/sub-102_space-T1_desc-masked_linreg_T2.nii.gz'
Rater_path = '/nasdata4/mjh/VALDO/tar/LACUNE/segmentation/cohort/Rater_mni/sub-102_space-T1_desc-Rater_linreg_Lacunes.nii.gz'

flair = nib.load(flair_path).get_fdata()
t1 = nib.load(t1_path).get_fdata()
t2 = nib.load(t2_path).get_fdata()
label = nib.load(Rater_path).get_fdata()

image_4d = np.stack([flair, t1, t2], axis = 0)
image_4d_tensor = torch.tensor(image_4d).float()
image_4d_tensor = torch.unsqueeze(image_4d_tensor, dim=0) # (1, 3, 182, 218, 182)
label_tensor = torch.tensor(label).float()
label_tensor = torch.unsqueeze(label_tensor, dim=0) # (1, 182, 218, 182)


# images = torch.randn(8, 3, 256, 256, 64)
# labels = torch.randint(0, 2, (8, 256, 256, 64))  # 바이너리 레이블

# 배치 크기 4로 크롭된 데이터 생성
batch_images, batch_labels = create_batch(image_4d_tensor, label_tensor, batch_size=1, crop_size=(64, 96, 64))

print("배치 이미지 크기:", batch_images.shape)  # (1, 3, 64, 96, 64)
print("배치 라벨 크기:", batch_labels.shape)  # (1, 64, 96, 64)

batch_images_np = batch_images[0, :, :, :, :].detach().cpu().numpy() # (3, 64, 96, 64)
batch_labels_np = batch_labels[0].detach().cpu().numpy() # (64, 96, 64)

for i in range(64):
    
    plt.subplot(8, 8, i+1)
    plt.imshow(batch_labels_np[:, :, i], cmap='gray')

plt.show()