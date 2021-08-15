import albumentations as albu
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

INPUT_SIZE = [320, 320]
CLASSES_SEG = 21


def get_seg_augmentation_train():
    return albu.Compose([
        albu.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
        albu.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.08, p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3,p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Flip(p=0.5),
        albu.Normalize(),
    ])


def get_seg_augmentation_val():
    return albu.Compose([
        albu.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
        albu.RandomRotate90(p=0.5),
        albu.Flip(p=0.5),
        albu.Normalize(),
    ])


class PASCAL_dataset_segmentation(Dataset):
    def __init__(self, img_root, mask_root, list_file, augmentation=None, img_suffix='.jpg', mask_suffix='.png'):
        super(PASCAL_dataset_segmentation, self).__init__()
        self.img_root = img_root
        self.mask_root = mask_root
        self.list_file = list_file
        self.augmentation = augmentation
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        try:
            with open(list_file, 'r') as f:
                self.ids = f.read().strip('\n').split('\n')
        except Exception as e:
            print(f'error:: {e}')
        self.imgs = list(filter(lambda x: x.strip(self.img_suffix) in self.ids, os.listdir(img_root)))
        self.masks = list(filter(lambda x: x.strip(self.mask_suffix) in self.ids, os.listdir(mask_root)))
        return

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.img_root, self.imgs[item]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_root, self.masks[item]))
        mask = np.stack([mask[:,:,0]==i for i in range(CLASSES_SEG)])
        mask[mask==True] = 1
        mask[mask==False] = 0
        mask = mask.swapaxes(0,1).swapaxes(1,2).astype('float32')
        label = np.sum(mask,axis=(0,1))
        label[label>=1] = 1
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            image = sample['image']
            mask = sample['mask']
        image = torch.FloatTensor(image).permute(2,0,1)
        mask = torch.FloatTensor(mask).permute(2,0,1)
        label = torch.IntTensor(label)
        return image, mask, label