# Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression

ongoing...

Pytorch implementation of [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399). 

## Prerequisites
+ Python 3.6 +
+ Pytorch 1.1.0
+ Scipy 0.19.1
+ cv2 3.3.0

## Usage

First, download dataset(Currently 300W supported).

**300W**

URL : [300W](https://ibug.doc.ic.ac.uk/resources/300-W/)

1. download [part1] ~ [part2]
2. locate 300W images, pts files according to this structure

data
```
|-- 300W
|   |-- 01_Indoor
|   |-- 02_Ourdoor
```


To train a model with downloaded dataset:

    $ python train.py

To test single image result:

    $ python detect.py


## Results

## Training details

## Related works

## Reference
+ CoordConv reference :: [CoordConv](https://github.com/mkocabas/CoordConv-pytorch)
+ Stacked Hourglass reference :: [Stacked Hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass)
+ Similar repo :: [AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss)
