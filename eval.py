import torch
import torch.nn as nn
import torch.optim as optim
from datasets.face300w import face300W
from models.hourglass import hg
import cv2
import numpy as np
from utils.post_process import _nms
import matplotlib.pyplot as plt

#setting
vis = False

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

nmes = list()
fail_count = 0

for iteration,sample in enumerate(test_loader):
    im,hmap,M,pts = sample
    img = im.transpose(1,-1)

    out = model(img)
    out = out[-1].squeeze().detach()
    out = _nms(out)
    out = out.numpy()

    pred = np.array([ [np.argmax(out[i])%64, np.argmax(out[i])//64] for i in range(0,68)]).astype(np.float64)

    ind2 = list() #second largest index
    mask = [[-1,-1],[-1,0],[-1,1],[1,0],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    for i in range(0,68):
        mx,my = int(pred[i,0]), int(pred[i,1])
        out[i,my,mx] -= 10
        neigbormax = np.argmax(out[i, my-1: my+2, mx-1:mx+2])
        out[i,my,mx] += 10
        dy, dx = mask[neigbormax]
        ind2.append([mx+dx, my+dy])

    ind2 = np.array(ind2).astype(np.float64)
    pred = (pred*3+ind2)/4. #move a quarter pixel toward 2nd largest
    pred *= (256/64.)

    gt = pts.squeeze().numpy()

    left_eye = np.average(gt[36:42], axis=0)
    right_eye = np.average(gt[42:48], axis=0)
    norm_factor = np.linalg.norm(left_eye - right_eye) #distance of eye center

    nme = (np.sum(np.linalg.norm(pred - gt, axis=1)) / pred.shape[0])/norm_factor
    print(nme)
    nmes.append((np.sum(np.linalg.norm(pred - gt, axis=1)) / pred.shape[0])/norm_factor)

    if nme > 0.1:
        fail_count += 1

    if vis:
        plt.scatter(pred[:,0],pred[:,1],c='r',s=8) #pred
        plt.scatter(gt[:,0],gt[:,1],c='b',s=8) #gt
        plt.imshow(im.squeeze().numpy())
        plt.show()

print('NME: {:.6f}'.format(sum(nmes)/len(nmes)*100))
