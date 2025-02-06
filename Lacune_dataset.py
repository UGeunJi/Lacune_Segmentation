import numpy as np
import os
import torch.utils.data as data
import torch
import natsort as natsort
from torchvision import transforms as transforms
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
import cv2
from torch.utils.data import DataLoader

def normalize(array):
    
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = np.nan_to_num(array)
    
    return array

def normalize_z(array):
    
    mean = np.mean(array)
    std = np.std(array)
    array = (array - mean) / std
    
    return array

def padding(array, size=224):
    
    h, w = array.shape
    h_pad = int((size - h) / 2)
    w_pad = int((size - w) / 2)
    
    array_pad = np.pad(array, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=0)
    return array_pad

def crop_image(flair_mri, t1_mri, t2_mri, label):
    
    coords = np.argwhere(flair_mri)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped_flair_mri = flair_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_t1_mri = t1_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_t2_mri = t2_mri[y_min:y_max+1, x_min:x_max+1]
    cropped_Rater = label[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_flair_mri, cropped_t1_mri, cropped_t2_mri, cropped_Rater

def crop_image_only(img):
    
    coords = np.argwhere(img)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped_img = img[y_min:y_max+1, x_min:x_max+1]
    
    return cropped_img

class ToTensor:
    def __call__(self, mri):
        # mri, label = sample['image'], sample['label']
        if len(mri.shape) == 2:
            mri = np.expand_dims(mri, axis=-1)
        mri = mri.transpose((2, 0, 1))
        
        return torch.from_numpy(mri).float()
        
        # return {'image': torch.from_numpy(mri).float(),
        #         'label': torch.tensor(label).long()}

class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL Image
        image = transforms.Resize(self.size)(image)
        image = np.array(image).astype(np.float32) / 255.0  # Normalize back to 0-1
        return image

class Healthy_dataset(data.Dataset):
    
    def __init__(self, img_npy_path, transform=None):
        
        self.img_path = img_npy_path
        self.transform = transform
        
        self.mri_files = [file for file in natsort.natsorted(os.listdir(img_npy_path))]
    
    def __getitem__(self, index):
        
        mri_path = os.path.join(self.img_path, self.mri_files[index])
        mri = np.load(mri_path) # (512, 512, 3)
        
        # mri = normalize(mri)
        
        mri_cropped = crop_image_only(mri)
        mri_cropped = normalize(mri_cropped)
        # mri = torch.from_numpy(mri) # (3, 512, 512)
        
        if self.transform:
            mri_cropped = self.transform(mri_cropped)
        
        return mri_cropped
    
    def __len__(self):
        return len(self.mri_files)

class Lacune_dataset(data.Dataset):
    
    def __init__(self, flair_npy_path, t1_npy_path, t2_npy_path, Rater_npy_path, transform=None):
        
        self.flair_img_path = flair_npy_path
        self.t1_img_path = t1_npy_path
        self.t2_img_path = t2_npy_path
        self.Rater_path = Rater_npy_path
        self.transform = transform
        
        self.flair_mri_files = [file for file in natsort.natsorted(os.listdir(flair_npy_path))]
        self.t1_mri_files = [file for file in natsort.natsorted(os.listdir(t1_npy_path))]
        self.t2_mri_files = [file for file in natsort.natsorted(os.listdir(t2_npy_path))]
        self.label_files = [file for file in natsort.natsorted(os.listdir(Rater_npy_path))]
    
    def __getitem__(self, index):
        
        flair_mri_path = os.path.join(self.flair_img_path, self.flair_mri_files[index])
        t1_mri_path = os.path.join(self.t1_img_path, self.t1_mri_files[index])
        t2_mri_path = os.path.join(self.t2_img_path, self.t2_mri_files[index])
        label_path = os.path.join(self.Rater_path, self.label_files[index])
        
        flair_mri = np.load(flair_mri_path) # (512, 512)
        t1_mri = np.load(t1_mri_path) # (512, 512)
        t2_mri = np.load(t2_mri_path) # (512, 512)
        label = np.load(label_path) # (512, 512)
        
        label = np.where(label > 0.0, 1.0, 0.0)
        
        flair_mri_pad = padding(flair_mri)
        t1_mri_pad = padding(t1_mri)
        t2_mri_pad = padding(t2_mri)
        label_pad = padding(label)
        
        flair_mri_pad = normalize_z(flair_mri_pad)
        t1_mri_pad = normalize_z(t1_mri_pad)
        t2_mri_pad = normalize_z(t2_mri_pad)
        
        if self.transform:
            flair_mri_pad = self.transform(flair_mri_pad)
            t1_mri_pad = self.transform(t1_mri_pad)
            t2_mri_pad = self.transform(t2_mri_pad)
            label_pad = self.transform(label_pad)
        
        mri_3ch = [flair_mri_pad, t1_mri_pad, t2_mri_pad]
        mri_3ch = torch.cat(mri_3ch, dim=0)
        
        return mri_3ch, label_pad
    
    def __len__(self):
        return len(self.label_files)

class Lacune_1ch_dataset(data.Dataset):
    
    def __init__(self, flair_npy_path, Rater_npy_path, transform=None):
        
        self.flair_img_path = flair_npy_path
        self.Rater_path = Rater_npy_path
        self.transform = transform
        
        self.flair_mri_files = [file for file in natsort.natsorted(os.listdir(flair_npy_path))]
        self.label_files = [file for file in natsort.natsorted(os.listdir(Rater_npy_path))]
    
    def __getitem__(self, index):
        
        flair_mri_path = os.path.join(self.flair_img_path, self.flair_mri_files[index])
        label_path = os.path.join(self.Rater_path, self.label_files[index])
        
        flair_mri = np.load(flair_mri_path) # (512, 512)
        label = np.load(label_path) # (512, 512)
        
        label = np.where(label > 0.0, 1.0, 0.0)
        
        flair_mri_pad = padding(flair_mri)
        label_pad = padding(label)
        
        flair_mri = normalize_z(flair_mri)
        
        if self.transform:
            flair_mri_pad = self.transform(flair_mri_pad)
            label_pad = self.transform(label_pad)
        
        return flair_mri_pad, label_pad
    
    def __len__(self):
        return len(self.label_files)
    
# class Lacune_png_dataset(data.Dataset):
    
#     def __init__(self, img_path, gt_path, transform=None):
        
#         self.img_path = img_path
#         self.gt_path = gt_path
#         self.transform = transform
        
#         self.img_files = list_file(img_path, '.png')
#         self.gt_files = list_file(gt_path, '.png')
        
#         self.img_files = natsort.natsorted(self.img_files)
#         self.gt_files = natsort.natsorted(self.gt_files)
    
#     def __getitem__(self, index):
        
#         mri_img_path = os.path.join(self.img_path, self.img_files[index])
#         label_path = os.path.join(self.gt_path, self.gt_files[index])
        
#         img = cv2.imread(mri_img_path).transpose(2, 0, 1)
#         gt = cv2.imread(label_path, 0)
#         H, W = gt.shape
        
#         img = img / 255
#         # gt = gt / 255
#         gt = np.where(cv2.resize(gt, (W,H))/255 > 0, 1.0, 0.0)
        
#         return img, gt
    
#     def __len__(self):
#         return len(self.gt_files)
    

# train_lacune_path = '/nasdata4/mjh/Diffusion/1_Anomaly/codes/data/with_lacune/train/mri_cohort2_npy'
# train_label_path = '/nasdata4/mjh/Diffusion/1_Anomaly/codes/data/with_lacune/train/label_cohort2_npy'

# test_lacune_path = '/nasdata4/mjh/Diffusion/1_Anomaly/codes/data/with_lacune/test/mri_cohort2_npy'
# test_label_path = '/nasdata4/mjh/Diffusion/1_Anomaly/codes/data/with_lacune/test/label_cohort2_npy'

# test_transform = transforms.Compose([
#     ResizeTransform((192, 192)),
#     ToTensor(),
# ])

# train_dataset = Lacune_dataset(train_lacune_path, train_label_path, test_transform)
# test_dataset = Lacune_dataset(test_lacune_path, test_label_path, test_transform)

# train_loader = data.DataLoader(train_dataset, batch_size=210,

class Lacune_png_dataset(data.Dataset):
    
    def __init__(self, img_path, transform=None):
        
        self.img_path = img_path
        self.gt_path = img_path + '/gt'
        self.transform = transform
        
        self.img_files = [file for file in natsort.natsorted(list_file(self.img_path, '.png'))]
        self.gt_files = [file for file in natsort.natsorted(list_file(self.gt_path, '.png'))]
    
    def __getitem__(self, index):
        
        img_3ch_path = os.path.join(self.img_path, self.img_files[index])
        label_path = os.path.join(self.gt_path, self.gt_files[index])
        
        x_img = cv2.imread(img_3ch_path)
        y_img = cv2.imread(label_path, 0)
        
        x_img = x_img.transpose(2, 0, 1) # (3, 224, 224) -> [t2, t1, flair]
        x_img = x_img / 255
        y_img = convert_gt(y_img)
        y_img = np.expand_dims(y_img, axis=0) # (1, 224, 224)
        
        if self.transform:
            x_img = self.transform(torch.from_numpy(x_img))
            y_img = self.transform(torch.from_numpy(y_img))
        else:
            x_img = torch.from_numpy(x_img)
            y_img = torch.from_numpy(y_img)
        
        # x_img = x_img.numpy()
        # y_img = y_img[0, :, :].numpy()
        
        # x_img = x_img / 255
        # y_img = convert_gt(y_img)
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.gt_files)

class Lacune_dataset(data.Dataset):
    
    def __init__(self, img_path, gt_path, transform=None):
        
        self.img_path = img_path
        self.gt_path = gt_path
        self.transform = transform
    
    def __getitem__(self, index):
        
        x_img = cv2.imread(self.img_path[index])
        y_img = cv2.imread(self.gt_path[index], 0)
        
        x_img = x_img.transpose(2, 0, 1) # (3, 224, 224) -> [t2, t1, flair]
        x_img = x_img / 255
        y_img = convert_gt(y_img)
        y_img = np.expand_dims(y_img, axis=0) # (1, 224, 224)
        
        if self.transform:
            x_img = self.transform(torch.from_numpy(x_img))
            y_img = self.transform(torch.from_numpy(y_img))
        else:
            x_img = torch.from_numpy(x_img)
            y_img = torch.from_numpy(y_img)
        
        # x_img = x_img.numpy()
        # y_img = y_img[0, :, :].numpy()
        
        # x_img = x_img / 255
        # y_img = convert_gt(y_img)
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.img_path)