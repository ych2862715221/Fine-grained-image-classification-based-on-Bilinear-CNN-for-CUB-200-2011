# -*- coding: utf-8 -*-
"""
date       : 2024-6-24
author     : Chuanhua Yu
to do      : 数据集Dataset定义
"""
import os
import random
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)

## 数据集定义
class BirdDataset(Dataset):
    def __init__(self, data_dir, filelist, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param filelist: str, 图片路径与标签txt
        :param transform: torch.transform，数据预处理
        """
        self.filelist = filelist
        self.data_dir = data_dir
        self.image_labels = self._get_img_label()  # image_labels存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        imgpath, label = self.image_labels[index]
        img = Image.open(imgpath).convert('RGB') # 0~255

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.image_labels) == 0:
            raise Exception(f"\ndata_dir:{self.data_dir} is a empty dir! Please checkout your path to images!")
        return len(self.image_labels)

    def _get_img_label(self):
        image_labels = []
        datas = open(self.data_dir+self.filelist).readlines()
        for data in datas:
            imgpath, label = data.strip().split(' ')
            image_labels.append((os.path.join(self.data_dir+'images/', imgpath), int(label)))

        return image_labels


if __name__ == "__main__":
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # path_dir = os.path.join(BASE_DIR, "data", "CUB_200_2011")
    path_dir = './data/CUB_200_2011/'
    dset = BirdDataset(data_dir=path_dir, filelist="train_shuffle.txt", transform=None)
    print('训练集图像数量：', dset.__len__())
    dset = BirdDataset(data_dir=path_dir, filelist="val_shuffle.txt", transform=None)
    print('验证集图像数量：', dset.__len__())





