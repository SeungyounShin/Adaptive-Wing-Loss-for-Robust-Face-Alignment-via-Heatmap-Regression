from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2,os
from scipy import ndimage
from scipy.ndimage.morphology import grey_dilation
import math
from PIL import Image
from utils.image import draw_umich_gaussian,draw_boundary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class face300W(Dataset):

    def __init__(self, root):
        self.root = root
        entry_point = os.listdir(root)
        self.ptsPath = list()
        self.imgPath = list()
        self.crop_pad = 30
        for i in entry_point:
            files = os.listdir(root+"/"+i)
            for f in files:
                if(f[-3:]=="png"):
                    self.imgPath.append(i+'/'+f)
                elif(f[-3:]=="pts"):
                    self.ptsPath.append(i+'/'+f)
                else:
                    print("a wrong format file in dataset : ",f)
        self.ptsPath.sort()
        self.imgPath.sort()
        self.frame = True
        self.resize = 256

    def __getitem__(self, idx):
        imgPath = self.root + "/" + self.imgPath[idx]
        ptsPath = self.root + "/" + self.ptsPath[idx]
        img = plt.imread(imgPath)
        #print(imgPath)
        if(len(img.shape)==2):
            # gray to rgb
            img = img.reshape(img.shape[0],img.shape[1],1)
            img = np.repeat(img,3,axis=2)
        w,h,c = img.shape
        with open(ptsPath) as ptsf:
            rows = [rows.strip() for rows in ptsf][3:-1]
            if len(rows) != 68:
                print("points are not 68")
                return None
            tofloat = lambda lst: [float(i) for i in lst]
            rows = [tofloat(pair.split(' ')) for pair in rows]
            rows = np.array(rows)

        minx,maxx = rows[:,0].min(),rows[:,0].max()
        miny,maxy = rows[:,1].min(),rows[:,1].max()
        face_h = maxx-minx

        img = img[int(max(0,miny-face_h)):int(min(maxy+self.crop_pad,w)),
                  int(max(0,minx-self.crop_pad)):int(min(maxx+self.crop_pad,h)), : ]

        rows[:,1] -= max(0,miny-face_h)
        rows[:,0] -= max(0,minx-self.crop_pad)

        if(self.frame):
            csh = img.shape
            frame = np.zeros((max(csh[0],csh[1]),max(csh[0],csh[1]),3))
            frame_ctr = np.array([max(csh[0],csh[1])//2,max(csh[0],csh[1])//2])

            frame[math.ceil(frame_ctr[0]-csh[0]/2.):math.ceil(frame_ctr[0]+csh[0]/2.),
                  math.ceil(frame_ctr[1]-csh[1]/2.):math.ceil(frame_ctr[1]+csh[1]/2.),:] = img

            if(csh[1] != frame.shape[1]):
                #가로패딩
                rows[:,0] += (frame.shape[0]-csh[1])/2.
            else:
                #새로패딩
                rows[:,1] += (frame.shape[0]-csh[0])/2.


            if(self.resize != None):
                rows /= frame.shape[0]
                frame = cv2.resize(frame, dsize=(self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
                rows *= float(self.resize)

        hmap = np.zeros((68+1, 64, 64), dtype=np.float32)
        M = np.zeros((68+1, 64, 64), dtype=np.float32)
        for ind, xy in enumerate(rows):
            hmap[ind] = draw_umich_gaussian(hmap[ind], xy/256.*64, 7)
        hmap[-1] = draw_boundary(hmap[-1],np.clip((rows/256.*64).astype(np.int),0,63))

        for i in range(len(M)):
            M[i] = grey_dilation(hmap[i], size=(3,3))
        M = np.where(M>=0.5, 1, 0)

        return frame, hmap , M, rows

    def __len__(self):
        return len(self.imgPath)


if __name__=="__main__":
    import sys,random
    sys.path.insert(1, '/Users/seungyoun/Desktop/ML/PR/Adaptive_Wing_Loss_for_Robut_Face_Alignemnt_via_Heatmap_Regression/code')
    from utils.image import draw_umich_gaussian, draw_boundary

    dataset_path = "/Users/seungyoun/Desktop/ML/PR/Adaptive_Wing_Loss_for_Robut_Face_Alignemnt_via_Heatmap_Regression/code/data/300W"
    dataset = face300W(dataset_path)
    l = len(dataset)
    img,hmap,M,pts = dataset[random.randint(0,l-1)]
    #img,pts = dataset[70]
    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.imshow(cv2.resize(hmap[-1], dsize=(256, 256), interpolation=cv2.INTER_AREA),alpha=0.3)
    plt.subplot(3,1,2)
    plt.imshow(np.max(hmap[:68], axis=0))
    plt.subplot(3,1,3)
    plt.imshow(np.max(M[:68], axis=0))

    plt.show()
