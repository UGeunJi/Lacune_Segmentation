'''
Edited by Fang Chen [cfun.cqupt@outlook.com].
Please cite our paper as follows if you use this code:
@ARTICLE{9735292,
  author={Chen, Fang and Gao, Chenqiang and Liu, Fangcen and Zhao, Yue and Zhou, Yuxi and Meng, Deyu and Zuo, Wangmeng},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Local Patch Network with Global Attention for Infrared Small Target Detection}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TAES.2022.3159308}}
'''
import numpy as np
import os
import cv2
import torch
import torch.utils.data as data
import natsort as natsort
from torchvision import transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import torch.nn as nn
import torch.nn.functional as F

def list_file(path, imtype='.bmp', print_=False):
	count = 0
	namelist = []
	for filename in os.listdir(path):
		# print(os.path.splitext(filename))
		if os.path.splitext(filename)[1] == imtype:
			namelist.append(filename)
			count = count + 1
			#fp = open(dirname+os.sep+filename,'r')
			#print(len(fp.readlines())-1)
			#fp.close()
	if print_:
		print(count)
	return namelist


def bilinear_interpolate(source, scale=[2,2], pad=0.5):
	sour_shape = source.shape
	(sh, sw) = (sour_shape[-2], sour_shape[-1])
	padding = pad*np.ones((sour_shape[0], sour_shape[1], sh+1, sw+1))
	padding[:,:,:-1,:-1] = source

	(th, tw) = (round(scale[0]*sh), round(scale[1]*sw))
	# targ_shape = list(sour_shape)
	# targ_shape[-2] = th
	# targ_shape[-1] = tw
	# target = np.zeros(targ_shape)

	grid = np.array(np.meshgrid(np.arange(th), np.arange(tw)), dtype=np.float32)
	xy = np.copy(grid)
	xy[0] *= sh/th
	xy[1] *= sw/tw
	x = xy[0].flatten()
	y = xy[1].flatten()

	clip = np.floor(xy).astype(np.int32)
	cx = clip[0].flatten()
	cy = clip[1].flatten()

	f1 = padding[:,:,cx,cy]
	f2 = padding[:,:,cx+1,cy]
	f3 = padding[:,:,cx,cy+1]
	f4 = padding[:,:,cx+1,cy+1]


	a = cx+1-x
	b = x-cx
	c = cy+1-y
	d = y-cy


	fx1 = a*f1 + b*f2
	fx2 = a*f3 + b*f4
	fy = c*fx1 + d*fx2
	# print(np.min(source),np.max(source))
	# print(np.min(fy),np.max(fy))
	fy = fy.reshape(fy.shape[0],fy.shape[1],tw,th).transpose((0,1,3,2))
	return fy

def center(cnd):
	p_lst = np.array(cnd)
	xmin = np.min(p_lst[:,0])
	xmax = np.max(p_lst[:,0])
	ymin = np.min(p_lst[:,1])
	ymax = np.max(p_lst[:,1])
	return ((xmin+xmax)/2, (ymin+ymax)/2)

def center3D(cnd):
	p_lst = np.array(cnd)
	xmin = np.min(p_lst[:,0])
	xmax = np.max(p_lst[:,0])
	ymin = np.min(p_lst[:,1])
	ymax = np.max(p_lst[:,1])
	zmin = np.min(p_lst[:,2])
	zmax = np.max(p_lst[:,2])
	return ((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2)

def getConnectedDomain(mask):
	mask = np.copy(mask).astype(np.uint8)
	cnd = []
	num, labels = cv2.connectedComponents(mask)
	for n in range(1, num):
		p_lst = np.array(np.where(labels==n)).transpose(1,0)
		p_lst = [tuple(p) for p in p_lst]
		cnd.append(p_lst)
	return cnd

def getConnectedDomain3D(mask):
	mask = np.copy(mask).astype(np.uint8)
	cnd = []
	
	structure = np.ones((3, 3, 3), dtype=np.int16)
	labeled_array, num_features = ndimage.label(mask, structure)
	
	for n in range(1, num_features + 1):
		
		p_lst = np.array(np.where(labeled_array == n)).transpose(1, 0)
		cnd.append(p_lst.tolist())
	return cnd

############################################# convert ground truth #################################################################
def convert_gt(img:np.ndarray) -> np.ndarray:
	# 0，1二值化
	return np.where(img != 0, 1, 0)

############################################# Nonlinear Scale ##########################################################################
def scaleNonlinear(img, e):
	img = np.abs(img)
	minv = np.min(img)
	maxv = np.max(img)
	if maxv-minv<=1e-3:
		return np.zeros(img.shape)
	img = ((img-minv)/(maxv-minv))**(1/e)
	return img

############################################ Gaussian Filter ##########################################################################
def distance(p1,p2):
	p1 = np.copy(p1)
	p2 = np.copy(p2)
	p = np.array([p1[0]-p2[0], p1[1]-p2[1]])**2
	p_sum = np.sqrt(np.sum(p, axis=0))
	return p_sum

def distance3D(p1,p2):
	p1 = np.copy(p1)
	p2 = np.copy(p2)
	p = np.array([p1[0]-p2[0], p1[1]-p2[1]], p1[2]-p2[2])**2
	p_sum = np.sqrt(np.sum(p, axis=0))
	return p_sum

def calDist(M,N):
	D = np.zeros((M,N))
	XY = np.meshgrid(np.arange(M), np.arange(N))
	XX = XY[0].flatten()
	YY = XY[1].flatten()
	D[XX, YY] = distance([XX, YY], [M/2, N/2])
	return D

def GaussianKernel(M, N, sigma):
	D = calDist(M,N)
	kernel = 1e-5 + np.exp(-(D**2)/(2*sigma**2))
	return kernel

def FourierTransfer(img):
	img = np.copy(img)
	fft = np.fft.fft2(img)
	fft = np.fft.fftshift(fft)
	return fft

def FourierTransferInverse(fft):
	fft = np.copy(fft)
	ifft = np.fft.ifftshift(fft)
	ifft = np.fft.ifft2(ifft)
	return ifft

def FilterFrequency(img, kernel, reverse=False):
	img = np.copy(img)
	h,w = img.shape
	fft = FourierTransfer(img)
	if reverse:
		fft /= kernel
	else:
		fft *= kernel
	ifft = FourierTransferInverse(fft)
	real = np.real(ifft)
	return real

def GaussianFilter_Frequency(img, sigma, reverse=False):
	img = np.copy(img)
	h,w = img.shape
	kernel = GaussianKernel(h,w,sigma)
	fft = FourierTransfer(img)
	if reverse:
		# kernel+=0.1
		# kernel[np.where(kernel<0.2)] = 0
		fft *= 1-kernel
	else:
		fft *= kernel
	ifft = FourierTransferInverse(fft)
	real = np.real(ifft)
	return real

# ##### Augmentation ######
# import torch
# import torchvision.transforms as T
# import torchvision.transforms.functional as F
# import random
# import numpy as np
# from PIL import Image
# from scipy.ndimage import map_coordinates, gaussian_filter

# # 랜덤 크롭
# class RandomCrop:
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         i, j, h, w = T.RandomCrop.get_params(image, self.output_size)
#         image = F.crop(image, i, j, h, w)
#         label = F.crop(label, i, j, h, w)
#         return {'image': image, 'label': label}

# # 랜덤 회전
# class RandomRotation:
#     def __init__(self, degrees):
#         self.degrees = degrees

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         angle = T.RandomRotation.get_params(self.degrees)
#         image = F.rotate(image, angle)
#         label = F.rotate(label, angle)
#         return {'image': image, 'label': label}

# # 좌우 및 상하 반전
# class RandomFlip:
#     def __init__(self, prob_horizontal=0.5, prob_vertical=0.5):
#         self.prob_horizontal = prob_horizontal
#         self.prob_vertical = prob_vertical

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         if random.random() < self.prob_horizontal:
#             image = F.hflip(image)
#             label = F.hflip(label)
#         if random.random() < self.prob_vertical:
#             image = F.vflip(image)
#             label = F.vflip(label)
#         return {'image': image, 'label': label}

# # Elastic 변형 (Elastic Deformation)
# class ElasticDeformation:
#     def __init__(self, alpha=34, sigma=4):
#         self.alpha = alpha
#         self.sigma = sigma

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         image = self.elastic_transform(image)
#         label = self.elastic_transform(label)
#         return {'image': image, 'label': label}

#     def elastic_transform(self, image):
#         image_np = np.array(image)  # PIL 이미지를 numpy 배열로 변환
#         shape = image_np.shape

#         dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
#         dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

#         x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#         indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

#         distored_image = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
#         return Image.fromarray(distored_image)

# # 랜덤 스케일링 (Zoom in/out)
# class RandomScaling:
#     def __init__(self, scale_range=(0.8, 1.2)):
#         self.scale_range = scale_range

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         scale = random.uniform(self.scale_range[0], self.scale_range[1])
#         image = F.resize(image, [int(scale * s) for s in image.size[::-1]])  # H x W
#         label = F.resize(label, [int(scale * s) for s in label.size[::-1]])

#         # 원래 크기로 센터 크롭 (scaling 후 다시 cropping)
#         image = F.center_crop(image, (224, 224))
#         label = F.center_crop(label, (224, 224))
        
#         return {'image': image, 'label': label}

def make_density(y_data, im_size, gaussian_std=5):
    
    # y_data : (B, 1, 224, 224) tensor
    y_data_len = y_data.shape[0]
    
    y_data_npy = y_data.detach().cpu().numpy()
    
    y_density = []
    for i in range(y_data_len):
        
        y_data_npy_ith = y_data_npy[i, 0, :, :]
        h, w = y_data_npy_ith.shape
        if np.all(y_data_npy_ith == 0):
            LPF = y_data_npy_ith
            LPF = np.nan_to_num(LPF)
        else:
            gaussianKernel = GaussianKernel(h, w, gaussian_std)
            LPF = FilterFrequency(y_data_npy_ith, gaussianKernel, False)
            LPF = scaleNonlinear(LPF, 1)
            LPF = LPF / np.sum(LPF)
            LPF = np.nan_to_num(LPF)
        y_density.append(LPF)
    
    y_density = np.array(y_density).reshape(-1, 1, im_size[0], im_size[1])
    # y_density = np.array(y_density).reshape(-1, 1, im_size[0]*im_size[1])
    
    return y_density

class weighted_BCE(nn.Module):
    
    def __init__(self):
        super(weighted_BCE, self).__init__()
        # 학습 가능한 파라미터로 클래스 가중치 설정 (초기값: 1.0)
        self.weight = nn.Parameter(torch.tensor([1.0, 1.0], requires_grad=True))

    def forward(self, inputs, targets):
        # weight[0]은 배경, weight[1]은 전경 가중치
        weights = self.weight[0] * (targets == 0) + self.weight[1] * (targets == 1)
        
        # BCE loss 계산 (weight 적용)
        loss = F.binary_cross_entropy(inputs, targets, weight=weights)
        return loss