import cv2
from math import *
import os
import sys
import os.path
from PIL import Image
import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from utils import *
import shutil
import math
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric
import natsort

class Evaluator:
    
    def __init__(self, pred_list, gt_list, tar_area, is_print=False):
        
        self.pred_list = pred_list
        self.gt_list = gt_list
        self.tar_area = tar_area
        assert len(self.tar_area) == 2
        self.is_print = is_print
        
    def target_metrics(self):
        
        is_print = self.is_print
        selected_gt_lst = []
        selected_pred_lst = []
            
        for idx in range(len(self.pred_list)):
            
            gt = self.gt_list[idx].detach().cpu().numpy()
            H,W = gt.shape
            gt_cds = getConnectedDomain(255*gt)
            # gt = np.where(cv2.resize(gt, (W,H))/255 > 0, 1.0, 0.0) # (224, 224)
            
            pred = self.pred_list[idx].detach().cpu().numpy() # (224, 224)
            
            cnds = getConnectedDomain(255*gt)

            for cnd in cnds:
                if len(cnd) >= self.tar_area[0] and len(cnd) <= self.tar_area[1]:
                    selected_gt_lst.append(gt)
                    selected_pred_lst.append(pred)
                    break
        
        true_detection = 0
        all_target = 0
        false_detection = 0
        TP = 0
        TP1 = 0
        FP = 0
        FN = 0
        
        for i in range(len(selected_pred_lst)):
            
            pred = selected_pred_lst[i] # (224, 224)
            # pred = np.where(pred >= threshold, 1.0, 0.0)
            H,W = pred.shape
            
            gt = selected_gt_lst[i]
            # gt = np.where(cv2.resize(gt, (W,H))/255 > 0, 1.0, 0.0)
            
            pred_cds = getConnectedDomain(255*pred)
            gt_cds = getConnectedDomain(255*gt)
            
            for pred_cd in pred_cds:
                if len(gt_cds)==0:
                    false_detection += 1
                    break
                for i,gt_cd in enumerate(gt_cds):
                    inters = list(set(pred_cd).intersection(set(gt_cd)))
                    
                    pred_center = center(pred_cd)
                    gt_center = center(gt_cd)
                    
                    if len(inters) > 0 and distance(pred_center, gt_center) <= 4:
                        
                        true_detection += 1
                        break
            
                else:
                    false_detection += 1
            
            for gt_cd in gt_cds:
                all_target += 1
                
                if len(pred_cds)==0:
                    FN += 1
                    break
                for i,pred_cd in enumerate(pred_cds):
                    inters = list(set(gt_cd).intersection(set(pred_cd)))
                    
                    gt_center = center(gt_cd)
                    pred_center = center(pred_cd)
                    
                    if len(inters) > 0 and distance(gt_center, pred_center) <= 4:
                        TP1 += 1
                        break
                
                else:
                    FN += 1
        
        Pd = true_detection/all_target
        Fa = false_detection/len(self.pred_list)
        
        FP = false_detection
        TP = true_detection
        
        if is_print:
            print('TP:',TP,'\tTP1:',TP1,'\tFP:',FP,'\tFN:',FN,'\n')
        
        if (TP1+FP) == 0:
            Prec = np.nan
        else:
            Prec = TP1/(TP1 + FP)
        if (TP1+FN) == 0:
            Rec = np.nan
        else:
            Rec = TP1/(TP1+FN)
        if (Prec+Rec) == 0:
            F1 = np.nan
        else:
            F1 = (2*Prec*Rec)/(Prec+Rec)
        
        if is_print:
            print('Pd:',Pd,'\tFa:',Fa,'\tF1:',F1,'\n')
        return Pd, Fa, Prec, Rec, F1
    
    def pixel_metrics(self):
        
        is_print = self.is_print
        selected_gt_lst = []
        selected_pred_lst = []
        
        for idx in range(len(self.pred_list)):
            
            gt = self.gt_list[idx].detach().cpu().numpy()
            H,W = gt.shape
            gt_cds = getConnectedDomain(255*gt)
            # gt = np.where(cv2.resize(gt, (W,H))/255 > 0, 1.0, 0.0) # (224, 224)
            
            pred = self.pred_list[idx].detach().cpu().numpy() # (224, 224)
            
            cnds = getConnectedDomain(255*gt)

            for cnd in cnds:
                if len(cnd) >= self.tar_area[0] and len(cnd) <= self.tar_area[1]:
                    selected_gt_lst.append(gt)
                    selected_pred_lst.append(pred)
                    break
        
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        pxls = 0
        
        for i in range(len(selected_pred_lst)):
            
            pred = selected_pred_lst[i] # (224, 224)
            # pred = np.where(pred >= threshold, 1.0, 0.0)
            H,W = pred.shape
            pxls += H*W
            
            gt = selected_gt_lst[i]
            # gt = np.where(cv2.resize(gt, (W,H))/255 > 0, 1.0, 0.0)
            
            tp += np.sum(np.where((pred+gt)==2, 1, 0))
            fn += np.sum(np.abs(pred-gt)*gt)
            fp += np.sum(np.abs(pred-gt)*(1-gt))
            tn += np.sum(np.where((pred+gt)==0, 1, 0))
        
        TP = tp/pxls
        FN = fn/pxls
        FP = fp/pxls
        TN = tn/pxls
        
        Prec = TP/(TP+FP)
        Rec = TP/(TP+FN)
        F1 = (2*Prec*Rec)/(Prec+Rec)
        
        if is_print:
            print('Prec:',Prec,'\tRec:',Rec,'\tF1:',F1,'\n')
            
        return Prec, Rec, F1