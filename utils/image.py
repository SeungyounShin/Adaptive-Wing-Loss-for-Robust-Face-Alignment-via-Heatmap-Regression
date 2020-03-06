from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def drawline(img,annots):
    shape = img.shape
    points_cnt = annots.shape[0]
    heatmap = np.zeros((shape[0],shape[1]))
    
    heatmap[[list(annots[:,1]),list(annots[:,0])]] = 1
    """
    for p in range(points_cnt-1):
        heatmap = cv2.line(heatmap,tuple(annots[p,:]),tuple(annots[p+1,:]),(255,255,255),thickness=5)

    heatmap = heatmap.astype(np.float32)
    annots = annots.astype(np.int)
    heatmap = cv2.polylines(heatmap, [annots], False, (255,255,255), thickness=3, lineType=cv2.LINE_AA)
    """

    annots = annots.astype(np.float64)
    x,y = annots[:,0],annots[:,1]
    l=len(x)
    t=np.linspace(0,1,l-2,endpoint=True)
    t=np.append([0,0,0],t)
    t=np.append(t,[1,1,1])
    tck=[t,[x,y],3]
    u3=np.linspace(0,1,(max(l*2,500)),endpoint=True)
    out = interpolate.splev(u3,tck)
    heatmap[[list(out[1].astype(np.int)),list(out[0].astype(np.int))]] = 1

    return heatmap

def draw_boundary(img,annot):
    heatmap = np.zeros((64,64))

    heatmap += drawline(img,annot[0:17,:])
    heatmap += drawline(img,annot[17:22,:])
    heatmap += drawline(img,annot[22:27,:])
    heatmap += drawline(img,annot[36:40,:])
    heatmap += drawline(img, np.array([annot[i] for i in [36, 41, 40, 39]]))
    heatmap += drawline(img,annot[42:46,:])
    heatmap += drawline(img, np.array([annot[i] for i in [42, 47, 46, 45]]))
    heatmap += drawline(img,annot[27:31,:])
    heatmap += drawline(img,annot[31:36,:])
    heatmap += drawline(img,annot[48:55,:])
    heatmap += drawline(img, np.array([annot[i] for i in [60, 61, 62, 63, 64]]))
    heatmap += drawline(img, np.array([annot[i] for i in [48, 59, 58, 57, 56, 55, 54]]))
    heatmap += drawline(img, np.array([annot[i] for i in [60, 67, 66, 65, 64]]))
    heatmap = np.clip(heatmap,0,1).astype(np.uint8)*255

    heatmap = 255-heatmap
    dist_transform = cv2.distanceTransform(heatmap, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_transform = dist_transform.astype(np.float64)
    sigma = 1
    gt = np.where(dist_transform < 3*sigma, np.exp(-(dist_transform*dist_transform)/(2*sigma*sigma)), 0 )

    return gt
