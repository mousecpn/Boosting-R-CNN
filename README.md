# Boosting R-CNN: Reweighting R-CNN Samples by RPN’s Error for Underwater Object Detection

This repository contains the code (in PyTorch) for the paper: 

## Introduction

Complicated underwater environments bring new challenges to object detection, such as unbalanced light conditions, low contrast, occlusion, and mimicry of aquatic organisms. Under these circumstances, the objects captured by the underwater camera will become vague, and the generic detectors often fail on these vague objects. This work aims to solve the problem from two perspectives: uncertainty modeling and hard example mining. We propose a two-stage underwater detector named boosting R-CNN, which comprises three key components. First, a new region proposal network named RetinaRPN is proposed, which provides high-quality proposals and considers objectness and IoU prediction for uncertainty to model the object prior probability. Second, the probabilistic inference pipeline is introduced to combine the first-stage prior uncertainty and the second-stage classification score to model the final detection score. Finally, we propose a new hard example mining method named boosting reweighting. Specifically, when the region proposal network miscalculates the object prior probability for a sample, boosting reweighting will increase the classification loss of the sample in the R-CNN head during training, while reducing the loss of easy samples with accurately estimated priors. Thus, a robust detection head in the second stage can be obtained. During the inference stage, the R-CNN has the capability to rectify the error of the first stage to improve the performance. Comprehensive experiments on two underwater datasets and two generic object detection datasets demonstrate the effectiveness and robustness of our method. 

![pipeline](C:\Users\Dell\Desktop\Boosting-R-CNN-Reweighting-R-CNN-Samples-by-RPN-s-Error-for-Underwater-Object-Detection\resources\pipeline.png)



## Dependencies

- Python==3.7.6
- PyTorch==1.0
- mmdetection==2.17.0
- mmcv==1.3.8
- numpy==1.16.3

## Installation

The basic installation follows with [mmdetection](https://github.com/mousecpn/mmdetection/blob/master/docs/get_started.md). It is recommended to use manual installation. 

## Datasets

**UTDAC2020**: https://drive.google.com/file/d/1avyB-ht3VxNERHpAwNTuBRFOxiXDMczI/view?usp=sharing

After downloading all datasets, create UTDAC2020 document.

```
$ cd data
$ mkdir S-UODAC2020
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

## Results

![qualitative results](C:\Users\Dell\Desktop\Boosting-R-CNN-Reweighting-R-CNN-Samples-by-RPN-s-Error-for-Underwater-Object-Detection\resources\qualitative results.png)

![video](C:\Users\Dell\Desktop\Boosting-R-CNN-Reweighting-R-CNN-Samples-by-RPN-s-Error-for-Underwater-Object-Detection\resources\video.gif)

## Acknowledgement

Thanks MMDetection team for the wonderful open source project!