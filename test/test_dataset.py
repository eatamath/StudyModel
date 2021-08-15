import sys
sys.path.append(r'D:\Code\github\eyisheng')

from dataset.PASCAL import *
import cv2
import matplotlib.pyplot as plt


def test_PASCAL_dataset_segmentation():
    img_root = r'D:\File\Dataset\PascalVOC2012\VOC2012\JPEGImages'
    mask_root = r'D:\File\Dataset\PascalVOC2012\VOC2012\SegmentationClass'
    list_file = r'D:\File\Dataset\PascalVOC2012\VOC2012\ImageSets\Segmentation\train.txt'
    dataset = PASCAL_dataset_segmentation(
        img_root=img_root,
        mask_root=mask_root,
        list_file=list_file,
        augmentation=get_seg_augmentation_train()
    )
    return dataset


if __name__=='__main__':
    dataset = test_PASCAL_dataset_segmentation()
    idx = 1003
    print(dataset[idx][0].shape, dataset[idx][1].shape, dataset[idx][2].shape)
    for i in range(10):
        plt.imsave('./image/'+str(i)+'.png', dataset[idx][0].int().numpy().astype('uint8'))
