import glob
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_list = glob.glob(img_dir + "*.jpg")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):
        mat = cv2.imread(self.img_list[idx], cv2.IMREAD_UNCHANGED)
        b, g, r = cv2.split(mat)
        merged = cv2.merge([b, b, g, g, r, b, b, r])
        image = torch.tensor(merged).T / 255.0
        image = image.float()
        label = torch.clone(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

