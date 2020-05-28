'''
Custom Dataset to load unlabel data
'''
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import math
import random

class UnlabelDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop=True):
        super(UnlabelDataset, self).__init__()
        self.filelist = [file for file in os.listdir(data_dir) if ".JPG" in file]
        self.data_dir = data_dir
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        filename = self.filelist[idx]
        img = Image.open(os.path.join(self.data_dir, filename))
        if self.crop:
            img = self.crop_object(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, filename

    def crop_object(self, img):
        cv_img = np.array(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        topx, topy, bottomx, bottomy = x, y, x+w, y+h
        # Now crop
        out = img.copy()
        out = out.crop((topx, topy, bottomx, bottomy))
        return out

class LabelDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop=True, limit=40):
        super(LabelDataset, self).__init__()
        self.files = np.array([])
        for folder, subfolder, images in os.walk(data_dir):
            if folder == data_dir:
                continue
            paths = np.array([os.path.join(folder, img) for img in images if '.JPG' in img])
            num_img = limit if limit < paths.shape[0] else paths.shape[0]
            np.random.shuffle(paths)
            self.files = np.concatenate([self.files, paths[:num_img]])
        self.transform = transform
        self.crop = crop


    def __len__(self):
        return self.files.shape[0]

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        filename = path.split("/")[-1]
        label = int(path.split("/")[-2])
        if self.crop:
            img = self.crop_object(img)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label, filename

    def crop_object(self, img):
        cv_img = np.array(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        topx, topy, bottomx, bottomy = x, y, x+w, y+h
        # Now crop
        out = img.copy()
        out = out.crop((topx, topy, bottomx, bottomy))
        return out

class Triple(Dataset):
    def __init__(self, data_dir, transform=None, crop=True, limit=40):
        super(Triple, self).__init__()
        file_dict = {}
        self.triple = []

        for folder, subfolder, images in os.walk(data_dir):
            if folder == data_dir:
                continue
            class_name = int(folder.split("/")[-1])
            paths = np.array([os.path.join(folder, img) for img in images if '.JPG' in img])
            num_img = limit if limit < paths.shape[0] else paths.shape[0]
            np.random.shuffle(paths)
            file_dict[class_name] = paths[:num_img]

        for cur_class, imgs in file_dict.items():
            neg_class = list(file_dict.keys())
            neg_class.pop(cur_class)
            for idx, anchor in enumerate(imgs):
                rest = np.delete(imgs, idx) 
                positive = random.choice(rest)
                negative = random.choice(file_dict[random.choice(neg_class)])
                self.triple.append((anchor, positive, negative, cur_class))
                
        self.transform = transform
        self.crop = crop


    def __len__(self):
        return len(self.triple)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path, label = self.triple[idx]
        anchor = self.load_image(anchor_path)
        positive = self.load_image(pos_path)
        negative = self.load_image(neg_path)        
        return anchor, positive, negative, label
    
    def load_image(self, path):
        img = Image.open(path)
        if self.crop:
            img = self.crop_object(img)

        if self.transform:
            img = self.transform(img)
            
        return img

    def crop_object(self, img):
        cv_img = np.array(img)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        topx, topy, bottomx, bottomy = x, y, x+w, y+h
        # Now crop
        out = img.copy()
        out = out.crop((topx, topy, bottomx, bottomy))
        return out