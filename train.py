import os

import torch.nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset.PASCAL import *
from metrics import metrics
from models import *
import run
import logger
from settings import *
from torch.utils.tensorboard import SummaryWriter

ENV = 'win'

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
if ENV=='win':
    data_root = r'D:\File\Dataset\PascalVOC2012\VOC2012'
    img_root = os.path.join(data_root, 'JPEGImages')
    mask_root = os.path.join(data_root, 'SegmentationClass')
    list_file_train = os.path.join(data_root, r'ImageSets\Segmentation\train.txt')
    list_file_val = os.path.join(data_root, r'ImageSets\Segmentation\trainval.txt')
    list_file_test = os.path.join(data_root, r'ImageSets\Segmentation\val.txt')
    DEVICE = 'cpu'
elif ENV=='linux':
    data_root = r'~/Documents/datasets'
    img_root = os.path.join(data_root, 'JPEGImages')
    mask_root = os.path.join(data_root, 'SegmentationClass')
    list_file_train = os.path.join(data_root, r'ImageSets/Segmentation/train.txt')
    list_file_val = os.path.join(data_root, r'ImageSets/Segmentation/trainval.txt')
    list_file_test = os.path.join(data_root, r'ImageSets/Segmentation/val.txt')
    DEVICE = 'cuda'

dataset_train = PASCAL_dataset_segmentation(
    img_root=img_root,
    mask_root=mask_root,
    list_file=list_file_train,
    augmentation=get_seg_augmentation_train()
)
dataset_val = PASCAL_dataset_segmentation(
    img_root=img_root,
    mask_root=mask_root,
    list_file=list_file_val,
    augmentation=get_seg_augmentation_val()
)
dataloader_train = DataLoader(dataset_train,
                              batch_size=conf['batchsize']['train'],
                              shuffle=True,
                              )

dataloader_val = DataLoader(dataset_val,
                            batch_size=conf['batchsize']['train'],
                            shuffle=True,
                            )

loss = torch.nn.BCELoss()
model = MyResNet50(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


train_epoch = run.TrainEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = run.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

logger_train = logger.MyLogger(stage='train')
logger_val = logger.MyLogger(stage='valid')
writer = SummaryWriter(LOG_PATH)

for epoch in range(30):
    train_logs = train_epoch.run(dataloader_train)
    valid_logs = valid_epoch.run(dataloader_val)
    logger_train.append(train_logs)
    logger_val.append(valid_logs)

    logger.write_logs(writer, train_logs, 'train')
    logger.write_logs(writer, valid_logs, 'valid')

    if epoch%SAVING_INTERVAL==1 or epoch==29:
        torch.save(model, './data/model-'+str(epoch)+'.pth')
