import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.data import DataLoader
from dataset.classification_dataset import *
from metrics import metrics
from models import *
import run
import logger
from settings import *

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

data_root = r'D:\File\Dataset\train'
dataset_test = MyDataset_Classification(root=data_root, folder_img='test', transform=get_simple_transformation())
dataloader_test = DataLoader(dataset_test,
                             batch_size=conf['batchsize']['test'],
                             shuffle=False)

loss = torch.nn.BCELoss()
model = MyResNet50(pretrained=False)
model = torch.load('') ### model path

valid_epoch = run.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

logger_test = logger.MyLogger(stage='test')

for epoch in range(1):
    valid_logs = valid_epoch.run(dataloader_test)
    logger_test.append(valid_logs)
