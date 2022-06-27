# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn as nn

from .base import BaseDetector
from .domain_classifier import *
import numpy as np
# from tools.BST_model import *
from tools.WaterTransfer import *
# from tools.AdaIN_ST import *
# from tools.MST import *
from collections import OrderedDict
import torch.distributed as dist
from ..builder import DETECTORS, build_backbone, build_head, build_neck
import itertools
import math

@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


@DETECTORS.register_module()
class DGFasterRCNN(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DGFasterRCNN, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.D = domain_cls(512,num_domains=2).cuda()
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3)
        self.count = 0.0
        self.total_img = 112128/2
        self.loss_domain = torch.nn.CrossEntropyLoss()

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(DGFasterRCNN, self).init_weights()

    def extract_feat(self, img,train=False):
        """Directly extract features from the backbone+neck."""
        self.count += img.shape[0]
        p = self.count / self.total_img
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        x = self.backbone(img)
        self.feature = x[3]
        if train == True:
            rev_feat = ReverseLayerF.apply(x[1],alpha)
            d_pred = self.D(rev_feat)
        if self.with_neck:
            x = self.neck(x)
        if train == True:
            return x, d_pred
        else:
            return x

    def extract_feat_crossgrad(self, img,train=False,detach = False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if train == True:
            if detach == True:
                d_pred = self.D(x[1].detach())
            else:
                d_pred = self.D(x[1])
        if self.with_neck:
            x = self.neck(x)
        if train == True:
            return x, d_pred
        else:
            return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      domain_label,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 0.1)
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()
        #################################    CrossGrad    #############################################


        # alpha = 0.2
        # device = img.device
        # img.requires_grad = True
        #
        # # img_l
        # loss_scalar = self.grad_forward(img,
        #                                 img_metas,
        #                                 gt_bboxes,
        #                                 gt_labels,
        #                                 gt_bboxes_ignore=gt_bboxes_ignore,
        #                                 gt_masks=gt_masks,
        #                                 proposals=proposals,
        #                                 **kwargs)
        # loss_scalar.backward(retain_graph=True)
        # clip = torch.tensor(20 / 255.).to(device)
        # img_l = img + self.clip_by_tensor(img.grad, -clip, clip)
        # img.grad *= 0
        #
        # # img_d
        # _, d_pred = self.extract_feat_crossgrad(img, train=True)
        # domain_label = domain_label.to(device)
        # domain_loss = self.loss_domain(d_pred, domain_label)
        # domain_loss.backward(retain_graph=True)
        # img = img + self.clip_by_tensor(img.grad, -clip, clip)
        # # img.grad *= 0
        # # img = img_d
        #
        # self.D.zero_grad()
        # self.backbone.zero_grad()
        # self.roi_head.zero_grad()
        # self.rpn_head.zero_grad()
        # self.neck.zero_grad()
        #
        # _, d_pred = self.extract_feat_crossgrad(img_l, train=True, detach = True)
        # domain_loss_turb = self.loss_domain(d_pred, domain_label)
        # x = self.extract_feat_crossgrad(img)
        # losses = dict()

        #################################    CrossGrad    #############################################

        #################################    DANN    #############################################
        x, d_pred = self.extract_feat(img, train=True)
        style_ids = torch.argmax(domain_label, axis=1)
        if domain_label.sum() != domain_label.shape[0]:
            print()
        domain_loss = self.loss_domain(d_pred, style_ids)
        losses = dict()
        #################################    DANN    #############################################

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        #################################    CrossGrad    #############################################

        # losses = self.rescale_loss(losses, alpha)
        # losses['domain_loss'] = domain_loss * (1 - alpha) + domain_loss_turb * alpha
        # losses['original loss'] = loss_scalar * (1 - alpha)

        #################################    CrossGrad    #############################################

        #################################    DANN    #############################################
        losses['domain_loss'] = domain_loss * 0.1
        # domain_loss.backward(retain_graph=True)

        #################################    DANN    #############################################
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def grad_forward(self,
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
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        loss_scalar = self.parse_losses(losses)[0]
        return loss_scalar

    def parse_losses(self,losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def clip_by_tensor(self,t, t_min, t_max):
        t = t.float()
        t_min = t_min.float()
        t_max = t_max.float()

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def rescale_loss(self,losses,alpha):
        for loss_name, loss_value in losses.items():
            if loss_name == 'acc':
                continue
            if isinstance(loss_value, list):
                for i in range(len(loss_value)):
                    loss_value[i] *= alpha
            else:
                losses[loss_name] *= alpha
        return losses


@DETECTORS.register_module()
class JiGENFasterRCNN(DGFasterRCNN):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(JiGENFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.D = jig_cls(2048,jig_classes = 31).cuda()
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3)
        self.count = 0.0
        self.total_img = 56064
        self.loss_jig = torch.nn.BCELoss()

        self.init_weights(pretrained=pretrained)

    def extract_feat(self, img,train=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if train == True:
            j_pred = self.D(x[3])
        if self.with_neck:
            x = self.neck(x)
        if train == True:
            return x, j_pred
        else:
            return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      img_puzzle,
                      jig_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x[0].shape = [2, 256, 192, 336]

        #################################    JiGEN_faster_rcnn_uodac    #############################################
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()
        x = self.extract_feat(img, train=False)
        _, jig_pred = self.extract_feat(img_puzzle, train=True)
        jig_loss = self.loss_jig(jig_pred, jig_labels)
        losses = dict()
        #################################    JiGEN_faster_rcnn_uodac    #############################################

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        #################################    JiGEN_faster_rcnn_uodac    #############################################
        losses['jig_loss'] = jig_loss * 0.1

        #################################    JiGEN_faster_rcnn_uodac    #############################################
        return losses


@DETECTORS.register_module()
class DGaugFasterRCNN(DGFasterRCNN):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DGaugFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        # self.WT = AdaIN_ST(vgg_path="/home/dailh/pytorch-AdaIN/models/vgg_normalised.pth",
        #               decoder_path="/home/dailh/pytorch-AdaIN/models/decoder.pth")
        self.WT = WaterTransfer(
            model_path="/home/dailh/Joint-Bilateral-Learning/checkpoints/epoch_6style_7.pth",
            style_num=14)
        # self.WT = MST(
        #     model_path = "/home/dailh/pytorch-multiple-style-transfer-master/output/epoch_10.model",
        #     style_num=7
        # )

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_mixup(self, img1, img2):
        """Directly extract features from the backbone+neck."""
        x, cont_loss = self.backbone(img1, img2, train=True)
        if self.with_neck:
            x = self.neck(x)
        return x, cont_loss

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      domain_label,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        def denormalize(img):
            mean = torch.tensor([123.675, 116.28, 103.53]).cuda()
            std = torch.tensor([58.395, 57.12, 57.375]).cuda()
            img = img * std.view(3, 1, 1).expand(img.shape)
            img = img + mean.view(3, 1, 1).expand(img.shape)
            img /= 255.0
            return img
        style_ids = torch.argmax(domain_label, axis=1)

        img_aug = self.WT.loop_forward(img, style_ids, img_size=512)
        # save_image(denormalize(img[0]), 'x1.jpg')
        # save_image(denormalize(img_aug[0]),'x2.jpg')
        p = np.random.random()
        if p < 0.5:
            x, cont_loss = self.extract_feat_mixup(img_aug, img)
        # elif p <= 0.6:
        #    x = self.extract_feat(img)
        else:
            x = self.extract_feat(img_aug)

        losses = dict()
        # losses['cont_loss'] = cont_loss
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses



@DETECTORS.register_module()
class MMDAAEFasterRCNN(DGFasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MMDAAEFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      domain_label,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        style_ids = torch.argmax(domain_label,axis=1)


        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,style_ids,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

@DETECTORS.register_module()
class EMAFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 k = 64):
        super(EMAFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        # self.backbone = build_backbone(backbone)
        #
        # if neck is not None:
        #     self.neck = build_neck(neck)
        #
        # if rpn_head is not None:
        #     rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        #     rpn_head_ = rpn_head.copy()
        #     rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        #     self.rpn_head = build_head(rpn_head_)
        #
        # if roi_head is not None:
        #     # update train and test cfg here for now
        #     # TODO: refactor assigner & sampler
        #     rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        #     roi_head.update(train_cfg=rcnn_train_cfg)
        #     roi_head.update(test_cfg=test_cfg.rcnn)
        #     self.roi_head = build_head(roi_head)
        #
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        # self.init_weights(pretrained=pretrained)
        # self.emau = nn.Sequential(EMAU(256,k=64), EMAU(256,k=64),EMAU(256,k=64),EMAU(256,k=64),EMAU(256,k=64))
        # self.svd = nn.Sequential(SVD_attention(256),SVD_attention(256),SVD_attention(256),SVD_attention(256),SVD_attention(256))
        self.emau = FP_EMAU(256, k=k)

    # def extract_feat(self, img, train=False):
    #     """Directly extract features from the backbone+neck."""
    #     x = self.backbone(img)
    #     mus = []
    #     outs = []
    #     if self.with_neck:
    #         x = self.neck(x)
    #     for i in range(len(x)):
    #         term,mu = self.emau[i](x[i])
    #         outs.append(term)
    #         mus.append(mu)
    #     if train == True:
    #         return tuple(outs), mus
    #     else:
    #         return tuple(outs)

    def extract_feat(self, img, train=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        mus = []
        outs = []
        if self.with_neck:
            x = self.neck(x)

        x, mu = self.emau(x)
        if train == True:
            return x, mu
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x,mu = self.extract_feat(img, train=True)


        with torch.no_grad():
            mu = mu.mean(dim=0, keepdim=True)
            momentum = 0.9
            self.emau.mu *= momentum
            self.emau.mu += mu * (1 - momentum)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)  # b * c * n
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x = mu.matmul(z_t)  # b * c * n
        x = x.view(b, c, h, w)  # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)

        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class FP_EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).
    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=3):
        super(FP_EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x)
        b, c, _, _ = x[0].size()
        idn = x.copy()
        for i in range(len(x)):
            x[i] = self.conv1(x[i])

        size = []
        for i in range(len(x)):
            size.append(x[i].shape)
            x[i] = x[i].view(b,c,-1)
        x_total = torch.cat(x, dim=-1)

        # The EM Attention
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x_total.permute(0, 2, 1)  # b * n * c
                z = torch.bmm(x_t, mu)  # b * n * k
                z = F.softmax(z, dim=2)  # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x_total, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * n
        x_total = mu.matmul(z_t)  # b * c * n
        x_total = x_total.view(b, c, -1)  # b * c * h * w
        x_total = F.relu(x_total, inplace=True)

        cur = 0
        for i in range(len(size)):
            term = x_total[:,:,cur:cur+size[i][-1]*size[i][-2]]
            cur += size[i][-1]*size[i][-2]
            x[i] = term.reshape(size[i])

        # The second 1x1 conv
        for i in range(len(x)):
            x[i] = self.conv2(x[i]) + idn[i]
            x[i] = F.relu(x[i], inplace=True)

        return tuple(x), mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

class SVD_attention(nn.Module):
    def __init__(self,indim, reduction = 4, k = 64, scale = 0.1):
        super(SVD_attention,self).__init__()
        self.embdim = indim//reduction
        self.k = k
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.scale = torch.nn.Parameter(torch.FloatTensor([scale]))
        self.conv1 = nn.Conv2d(indim, self.embdim, kernel_size=1)
        self.deconv1 = nn.Conv2d(self.embdim, indim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(indim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        res = self.pool(x)
        res = self.conv1(res)
        batch_size = x.shape[0]
        w = res.shape[-2]
        h = res.shape[-1]
        c = res.shape[1]
        feats_rec = []
        for i in range(batch_size):
            feat = res[i]
            featmap_flatten = feat.reshape(feat.shape[0], -1) # (c, W * H)
            u, _, _ = torch.svd(featmap_flatten)
            u = u[:, :self.k].transpose(0, 1) # (k, c)
            u = self._l2norm(u, dim=1)
            heatmap = torch.matmul(u, featmap_flatten) # (k, W * H)
            # weight = F.softmax(heatmap, dim=1).view(self.k, 1, w*h) # spatial softmax
            weight = F.softmax(heatmap, dim=0) # channel softmax
            feat_rec = torch.matmul(u.transpose(0, 1), weight).reshape(c, w, h)
            feats_rec.append(feat_rec)
        feats_rec = torch.stack(feats_rec)
        feats_rec = F.relu(feats_rec)
        res = self.deconv1(feats_rec)
        res = F.interpolate(res, size=(x.shape[-2],x.shape[-1]), mode='bilinear', align_corners=True)
        res = self.bn(res)
        # x = x * (1 - self.scale) + res * self.scale
        x = x + res
        x = F.relu(x)
        return x

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward2(self, x):
        res = self.pool(x)
        res = self.conv1(res)
        batch_size = x.shape[0]
        w = res.shape[-2]
        h = res.shape[-1]
        c = res.shape[1]

        featmap_flatten = res.permute(1, 0 ,2, 3).reshape(res.shape[1], -1) # (c, B * W * H)
        u, _, _ = torch.svd(featmap_flatten)
        u = u[:, :self.k].transpose(0, 1) # (k, c)
        u = self._l2norm(u, dim=1)
        heatmap = torch.matmul(u, featmap_flatten) # (k, B * W * H)
        # weight = F.softmax(heatmap, dim=1).view(self.k, 1, w*h) # spatial softmax
        weight = F.softmax(heatmap, dim=0) # channel softmax
        feats_rec = torch.matmul(u.transpose(0, 1), weight).reshape(c, batch_size, w, h).permute(1,0,2,3)
        feats_rec = F.relu(feats_rec)
        res = self.deconv1(feats_rec)
        res = F.interpolate(res, size=(x.shape[-2],x.shape[-1]), mode='bilinear', align_corners=True)
        # res = self.bn(res)
        # x = x * (1 - self.scale) + res * self.scale
        x = x + res
        x = F.relu(x)
        return x