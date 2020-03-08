# Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression


❗ongoing repo 



Pytorch implementation of [Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399). 

official implementation can be found at [official](https://github.com/protossw512/AdaptiveWingLoss). 

<p align="center"><img src="https://github.com/SeungyounShin/Adaptive-Wing-Loss-for-Robust-Face-Alignment-via-Heatmap-Regression/blob/master/assets/%E1%84%89%E1%85%B3%E1%86%BC%E1%84%8B%E1%85%B2%E1%86%AB%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA3.png?raw=true" alt="result" width="60%"></p>



## Prerequisites
+ Python 3.6 +
+ Pytorch 1.1.0
+ Scipy 0.19.1
+ cv2 3.3.0

## Usage

First, download dataset(Currently 300W supported).

**300W** [link](https://ibug.doc.ic.ac.uk/resources/300-W/)

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

## Model overview
<p align="center"><img src="assets/hourglass.png" alt="model" width="60%"></p>

## Results

## Training details

## evalutaion

## Reference
+ [CoordConv](https://github.com/mkocabas/CoordConv-pytorch)
+ [Stacked Hourglass](https://github.com/princeton-vl/pytorch_stacked_hourglass)
+ [AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss)
