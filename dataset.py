from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import math

class MyDataset(Dataset):
	def __init__(self, data_dir, transform=None, crop=True):
		super(MyDataset, self).__init__()
		# try:
		self.data_dir, self.transform, self.crop = data_dir, transform, crop
		filelist = [file for file in os.listdir(data_dir) if ".JPG" in file]
		self.data = [Image.open(os.path.join(data_dir, fname)) for fname in filelist]
		# except Exception:
		# 	raise("Cannot find the folder...")
		if self.crop:
			self.data = [self.crop_object(img) for img in self.data]

		if self.transform is not None:
			self.data = [self.transform(img) for img in self.data]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

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
