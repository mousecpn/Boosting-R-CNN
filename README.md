# Boosting R-CNN: Reweighting R-CNN Samples by RPN’s Error for Underwater Object Detection

This repository contains the code (in PyTorch) for the paper: ([arxiv: 2206.13728](http://arxiv.org/abs/2206.13728))

## Introduction

Complicated underwater environments bring new challenges to object detection, such as unbalanced light conditions, low contrast, occlusion, and mimicry of aquatic organisms. Under these circumstances, the objects captured by the underwater camera will become vague, and the generic detectors often fail on these vague objects. This work aims to solve the problem from two perspectives: uncertainty modeling and hard example mining. We propose a two-stage underwater detector named boosting R-CNN, which comprises three key components. First, a new region proposal network named RetinaRPN is proposed, which provides high-quality proposals and considers objectness and IoU prediction for uncertainty to model the object prior probability. Second, the probabilistic inference pipeline is introduced to combine the first-stage prior uncertainty and the second-stage classification score to model the final detection score. Finally, we propose a new hard example mining method named boosting reweighting. Specifically, when the region proposal network miscalculates the object prior probability for a sample, boosting reweighting will increase the classification loss of the sample in the R-CNN head during training, while reducing the loss of easy samples with accurately estimated priors. Thus, a robust detection head in the second stage can be obtained. During the inference stage, the R-CNN has the capability to rectify the error of the first stage to improve the performance. Comprehensive experiments on two underwater datasets and two generic object detection datasets demonstrate the effectiveness and robustness of our method. 

![pipeline](https://user-images.githubusercontent.com/46233799/175853966-7e9464aa-406b-42fa-a1b4-b0639dbaf577.png)


## Dependencies

- Python==3.7.6
- PyTorch==1.7
- mmdetection==2.17.0
- mmcv==1.4.0
- numpy==1.16.3

## Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to use manual installation. 

## Datasets

**UTDAC2020**: https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing

After downloading all datasets, create UTDAC2020 document.

```
$ cd data
$ mkdir UTDAC2020
```

It is recommended to symlink the dataset root to `$data`.

```
Boosting-R-CNN-Reweighting-R-CNN-Samples-by-RPN-s-Error-for-Underwater-Object-Detection
├── data
│   ├── UTDAC2020
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── annotations
```

**COCO**: https://cocodataset.org/#download

**PASCAL VOC**: http://host.robots.ox.ac.uk/pascal/VOC/

## Train

**UTDAC2020**

```
$ python tools/train.py configs/boosting_rcnn/boosting_rcnn_r50_pafpn_1x_utdac.py
```

**COCO**

```
$ python tools/train.py configs/boosting_rcnn/boosting_rcnn_r50_pafpn_1x_coco.py
```

**PASCAL VOC**

```
$ python tools/train.py configs/boosting_rcnn/boosting_rcnn_r50_pafpn_1x_voc.py
```

## Test

**UTDAC2020**

```
$ python tools/test.py configs/boosting_rcnn/boosting_rcnn_r50_pafpn_1x_utdac.py <path/to/checkpoints>
```

## Checkpoint

**UTDAC2020**: https://drive.google.com/file/d/1p5SQe0opW1CbCjUsRHlBfANBXdoJ_kOs/view?usp=sharing

**COCO**: https://drive.google.com/file/d/1uuE7GJvmY1pxW6okFgsX_khkYBy6PfB2/view?usp=sharing


## Results

<img width="572" alt="image" src="https://user-images.githubusercontent.com/46233799/191498834-26c4aeff-23ad-4168-adc7-59b1c5b6d87a.png">

<img width="565" alt="image" src="https://user-images.githubusercontent.com/46233799/191498896-2285fd70-c385-43e4-a0db-992f45d7d0c8.png">


![qualitative results](https://user-images.githubusercontent.com/46233799/175853981-828f55f8-d868-4524-ab8a-cc24ab92ec05.png)

![video 00_00_00-00_00_30](https://user-images.githubusercontent.com/46233799/175854144-e61280e7-9c06-4d59-b739-3e4974121dbb.gif)

## Acknowledgement

Thanks MMDetection team for the wonderful open source project!
