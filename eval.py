import torch
import torch.nn as nn
import torch.optim as optim
from datasets.face300w import face300W
from models.hourglass import hg
import cv2
import numpy as np

#dataset load
dataset_path = "./data/test"
test_dataset = face300W(dataset_path)
datalen = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                           shuffle=True, num_workers=0)
print("test images : ",datalen)

#model load
model = torch.load('./ckpt/120.pth', map_location='cpu')
#model = torch.load('./ckpt/WFLW_4HG.pth', map_location='cpu')
model.eval()
