from .two_stage import TwoStageDetector
import torch
import torch.nn as nn

from .base import BaseDetector
from .domain_classifier import *
import numpy as np
from collections import OrderedDict
import torch.distributed as dist
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import itertools
import math
from mmdet.core import multi_apply
from mmcv.cnn import ConvModule

@DETECTORS.register_module()
class ProbTwoStage(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 multiclsrpn=False):
        super(ProbTwoStage, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.fusion_convs = nn.ModuleList()
        for i in range(5):
            self.fusion_convs.append(
                ConvModule(
                    512,
                    256,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)))
        self.multiclsrpn = multiclsrpn

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.multiclsrpn:
                rpn_losses, rpn_results, rpn_feats = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
            else:
                rpn_losses, rpn_results, rpn_feats = self.rpn_head.forward_train(
                    x,
                    img_metas,
                    gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
            proposal_list, cls_scores, bbox_preds, iou_preds = rpn_results
            feats = []
            for i in range(len(x)):
                # feats.append(self.iou_attn(x[i],iou_preds[i]))
                feats.append(self.feature_fusion((x[i], rpn_feats[i]), self.fusion_convs[i]))
            x = feats
        else:
            proposal_list = proposals
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def iou_attn(self, x, iou_preds):
        attn,_ = torch.max(iou_preds, dim=1, keepdim=True)
        attn = attn.detach()
        x = x * attn.sigmoid() + x
        return x

    def feature_fusion(self,x, conv):
        x = torch.cat(x, dim=1)
        x = conv(x)
        return x

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            rpn_results, rpn_feats = self.rpn_head.simple_test_rpn(x, img_metas, bridge=True)
            proposal_list, cls_scores, bbox_preds, iou_preds = rpn_results
            feats = []
            for i in range(len(x)):
                # feats.append(self.iou_attn(x[i],iou_preds[i]))
                feats.append(self.feature_fusion((x[i], rpn_feats[i]), self.fusion_convs[i]))
            x = feats
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)