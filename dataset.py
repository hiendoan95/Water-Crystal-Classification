from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import cv2
import math

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
		img = Image.open(os.path.join(self.data_dir, self.filelist[idx]))
		if self.crop:
			img = self.crop_object(img)

		if self.transform is not None:
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
