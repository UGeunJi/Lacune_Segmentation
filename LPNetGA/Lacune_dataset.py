import numpy as np
import os
import torch.utils.data as data
import torch
import natsort as natsort
from torchvision import transforms as transforms
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
from utils.utils import *
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
        
        # x_img = x_img.numpy()
        # y_img = y_img[0, :, :].numpy()
        
        # x_img = x_img / 255
        # y_img = convert_gt(y_img)
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.gt_files)

class Lacune_fold_dataset(data.Dataset):
    
    def __init__(self, slices_list, transform=None):
        
        self.slices_list = slices_list
        self.transform = transform
    
    def __getitem__(self, index):
        
        flair_img, t1_img, t2_img, label_img = self.slices_list[index]
        
        flair_img = padding(flair_img)
        t1_img = padding(t1_img)
        t2_img = padding(t2_img)
        label_img = padding(label_img)
        
        flair_img = normalize(flair_img)
        t1_img = normalize(t1_img)
        t2_img = normalize(t2_img)
        
        flair_img = np.nan_to_num(flair_img)
        t1_img = np.nan_to_num(t1_img)
        t2_img = np.nan_to_num(t2_img)
        
        x_img = np.stack([flair_img, t1_img, t2_img], axis=0) # (3, 224, 224)
        x_img = (x_img * 255).astype(np.uint8)
        x_img = x_img / 255
        
        y_img = convert_gt(label_img)
        y_img = np.expand_dims(y_img, axis=0)
        
        if self.transform:
            x_img = self.transform(torch.from_numpy(x_img))
            y_img = self.transform(torch.from_numpy(y_img))
        else:
            x_img = torch.from_numpy(x_img)
            y_img = torch.from_numpy(y_img)
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.slices_list)

class Lacune_fold_raw_dataset(data.Dataset):
    
    def __init__(self, slices_list, transform=None):
        
        self.slices_list = slices_list
        self.transform = transform
    
    def __getitem__(self, index):
        
        flair_img, t1_img, t2_img, label_img = self.slices_list[index]
        
        flair_img = normalize(flair_img)
        t1_img = normalize(t1_img)
        t2_img = normalize(t2_img)
        
        flair_img = np.nan_to_num(flair_img)
        t1_img = np.nan_to_num(t1_img)
        t2_img = np.nan_to_num(t2_img)
        
        x_img = np.stack([flair_img, t1_img, t2_img], axis=0) # (3, 224, 224)
        x_img = (x_img * 255).astype(np.uint8)
        x_img = x_img / 255
        
        y_img = convert_gt(label_img)
        y_img = np.expand_dims(y_img, axis=0)
        
        if self.transform:
            x_img = self.transform(torch.from_numpy(x_img))
            y_img = self.transform(torch.from_numpy(y_img))
        else:
            x_img = torch.from_numpy(x_img)
            y_img = torch.from_numpy(y_img)
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.slices_list)

class Lacune_png_all_modal_dataset(data.Dataset):
    
    def __init__(self, img_path, transform=None):
        
        self.flair_path = img_path + '/flair'
        self.t1_path = img_path + '/t1'
        self.t2_path = img_path + '/t2'
        self.gt_path = img_path + '/gt'
        self.transform = transform
        
        self.flair_files = [file for file in natsort.natsorted(list_file(self.flair_path, '.png'))]
        self.t1_files = [file for file in natsort.natsorted(list_file(self.t1_path, '.png'))]
        self.t2_files = [file for file in natsort.natsorted(list_file(self.t2_path, '.png'))]
        self.gt_files = [file for file in natsort.natsorted(list_file(self.gt_path, '.png'))]
    
    def __getitem__(self, index):
        
        flair_path = os.path.join(self.flair_path, self.flair_files[index])
        t1_path = os.path.join(self.t1_path, self.t1_files[index])
        t2_path = os.path.join(self.t2_path, self.t2_files[index])
        label_path = os.path.join(self.gt_path, self.gt_files[index])
        
        x_flair_img = cv2.imread(flair_path, 0)
        x_t1_img = cv2.imread(t1_path, 0)
        x_t2_img = cv2.imread(t2_path, 0)
        y_img = cv2.imread(label_path, 0)
        
        x_img = np.stack([x_flair_img, x_t1_img, x_t2_img], axis=-1)
        
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
        
        return x_img, y_img
    
    def __len__(self):
        return len(self.gt_files)