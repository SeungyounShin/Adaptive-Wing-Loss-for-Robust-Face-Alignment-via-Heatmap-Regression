import torch
import matplotlib.pyplot as plt
import cv2,random
import numpy as np
from datasets.face300w import face300W
from utils.post_process import _nms

#model load
model = torch.load('./ckpt/NME(4.46).pth', map_location='cpu')
#model = torch.load('./ckpt/WFLW_4HG.pth', map_location='cpu')
model.eval()

#load image


path = "./data/real_world_test/6.png"
img = plt.imread(path)
img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)[:,:,:3]
"""
dataset_path = "/Users/seungyoun/Desktop/ML/PR/Adaptive_Wing_Loss_for_Robut_Face_Alignemnt_via_Heatmap_Regression/code/data/test"
dataset = face300W(dataset_path)
l = len(dataset)
#img,hmap,M,pts = dataset[random.randint(0,l-1)]
img,hmap,M,pts = dataset[200]
"""



#plt.subplot(2,1,1)

plt.imshow(img)

#infer
input_ = torch.tensor(img).transpose(0,-1).view(1,3,256,256)
out = model(input_)[-1].squeeze().detach()
#out = _nms(out)
out = out.numpy()

# (out.shape) 69,64,64
#plt.imshow(cv2.resize(np.max(out[:68], axis=0), dsize=(256, 256), interpolation=cv2.INTER_LINEAR),alpha=0.5)
#plt.imshow(cv2.resize(out[0], dsize=(256, 256), interpolation=cv2.INTER_LINEAR),alpha=0.5)
ind = np.array([ [np.argmax(out[i])%64,np.argmax(out[i])//64] for i in range(0,68)]).astype(np.float64)

ind2 = list() #second largest index
mask = [[-1,-1],[-1,0],[-1,1],[1,0],[0,0],[0,1],[1,-1],[1,0],[1,1]]
for i in range(0,68):
    mx,my = int(ind[i,0]), int(ind[i,1])
    out[i,my,mx] -= 10
    neigbormax = np.argmax(out[i, my-1: my+2, mx-1:mx+2])
    out[i,my,mx] += 10
    dy, dx = mask[neigbormax]
    ind2.append([mx+dx, my+dy])

ind2 = np.array(ind2).astype(np.float64)
new = (ind*3+ind2)/4.

ind *= (256/64.)
ind2 *= (256/64.)
new *= (256/64.)

#plt.scatter(ind[:,0],ind[:,1],c='r',s=3)
#plt.scatter(ind2[:,0],ind2[:,1],c='b',s=3)
plt.scatter(new[:,0],new[:,1],c='black',s=10)

#plt.subplot(2,1,2)
#plt.imshow(out[0],alpha=0.3)


plt.show()
