import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import batched_nms,DeformConv2dPack
from mmcv.runner import force_fp32
from mmdet.core.bbox.iou_calculators import bbox_overlaps
EPS = 1e-12
from ..builder import HEADS, build_loss
from .rpn_head import RPNHead
from mmdet.core import (anchor_inside_flags, unmap, multi_apply, reduce_mean, images_to_levels)
from mmcv.runner import BaseModule
from ..backbones.resnet import BasicBlock, Bottleneck
from ..backbones.res2net import Res2Layer, Bottle2neck
from ..backbones.resnest import Bottleneck as BottleSneck
from ..losses import GHMR
from ..losses import VarifocalLoss
import torch.nn.functional as F
from mmcv.cnn.bricks.norm import build_norm_layer

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class ASPP_share(nn.Module):
    def __init__(self, dilations, in_channels, channels, norm_cfg = None):
        super(ASPP_share, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.shared_conv = ConvModule(in_channels,channels,3)
        self.w = self.shared_conv.conv.weight
        self.b = self.shared_conv.conv.bias
        self.conv1x1 = nn.Conv2d(len(dilations)*channels,channels,1)
        self.norm_name, self.norm = build_norm_layer(norm_cfg, channels)
        # self.act = nn.ReLU(inplace=True)
        self.act = Mish()

    def forward(self, x):
        """Forward function."""
        bs,c,h,w = x.shape
        aspp_outs = []
        for dilation in self.dilations:
            aspp_outs.append(F.conv2d(x, self.w, self.b, stride=1, padding=dilation,dilation=dilation))

        aspp_outs = torch.cat(aspp_outs,dim=1)
        aspp_outs = self.conv1x1(aspp_outs)
        aspp_outs = self.act(aspp_outs)
        return aspp_outs

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = x * self.sigmoid(out)
        return x

class DCNModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 norm_cfg = None,
                 groups= 1,
                 deform_groups = 1,
                 bias = False):
        super(DCNModule, self).__init__()
        self.dcn = DeformConv2dPack(in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 groups= 1,
                 deform_groups = 1,
                 bias = False)
        self.norm_name, self.norm = build_norm_layer(norm_cfg, out_channels)
        # self.act = nn.ReLU(inplace=True)
        self.act = Mish()


    def forward(self, x):
        x = self.dcn(x)
        x = self.norm(x)
        x = self.act(x)
        return x


@HEADS.register_module()
class ATSSRPNHead(RPNHead):
    def __init__(self,
                 *args,
                 stacked_convs=4,
                 conv_cfg=None,
                 gamma=1,
                 atss=False,
                 bridge=False,
                 last_conv='norm',
                 aug_reg_loss=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness =dict(
                            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='rpn_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.last_conv = last_conv
        super(ATSSRPNHead, self).__init__(*args, init_cfg=init_cfg, **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        self.gamma = gamma
        self.atss = atss
        self.bridge = bridge
        # self.L1 = build_loss(dict(type='L1Loss', loss_weight=1.0))
        if aug_reg_loss is not None:
            self.with_aug_loss = True
            self.aug_loss = build_loss(aug_reg_loss)
        else:
            self.with_aug_loss = False


    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if i == self.stacked_convs - 1:
                if self.last_conv == 'dcn':
                    self.rpn_convs.append(
                        DCNModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            deform_groups=32))
                elif self.last_conv == 'aspp':
                    self.rpn_convs.append(ASPP_share(
                                (1, 3, 5, 7),
                                chn,
                                self.feat_channels,
                                norm_cfg=self.norm_cfg))
                else:
                    self.rpn_convs.append(
                        ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg))
            else:
                self.rpn_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.rpn_convs.append(CBAM(self.feat_channels))
            # self.rpn_convs.append(Mish())

        self.rpn_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.rpn_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.rpn_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

    def forward_single(self, x, scale, bridge):
        for conv in self.rpn_convs:
            x = conv(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = scale(self.rpn_reg(x)).float()
        rpn_iou_pred = self.rpn_iou(x)
        if bridge:
            return rpn_cls_score, rpn_bbox_pred, rpn_iou_pred, x
        return rpn_cls_score, rpn_bbox_pred, rpn_iou_pred


    # def _init_layers(self):
    #     """Initialize layers of the head."""
    #     self.relu = nn.ReLU(inplace=True)
    #     self.cls_convs = nn.ModuleList()
    #     self.reg_convs = nn.ModuleList()
    #     for i in range(self.stacked_convs):
    #         chn = self.in_channels if i == 0 else self.feat_channels
    #         self.cls_convs.append(
    #             ConvModule(
    #                 chn,
    #                 self.feat_channels,
    #                 3,
    #                 stride=1,
    #                 padding=1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg))
    #         self.reg_convs.append(
    #             ConvModule(
    #                 chn,
    #                 self.feat_channels,
    #                 3,
    #                 stride=1,
    #                 padding=1,
    #                 conv_cfg=self.conv_cfg,
    #                 norm_cfg=self.norm_cfg))
    #     self.rpn_cls = nn.Conv2d(
    #         self.feat_channels,
    #         self.num_anchors * self.cls_out_channels,
    #         3,
    #         padding=1)
    #     self.rpn_reg = nn.Conv2d(
    #         self.feat_channels, self.num_anchors * 4, 3, padding=1)
    #     self.rpn_iou = nn.Conv2d(
    #         self.feat_channels, self.num_anchors * 1, 3, padding=1)
    #     self.scales = nn.ModuleList(
    #         [Scale(1.0) for _ in self.anchor_generator.strides])

    # def forward_single(self, x, scale, bridge):
    #     cls_feat = x
    #     reg_feat = x
    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     rpn_cls_score = self.rpn_cls(cls_feat)
    #     # we just follow atss, not apply exp in bbox_pred
    #     rpn_bbox_pred = scale(self.rpn_reg(reg_feat)).float()
    #     rpn_iou_pred = self.rpn_iou(reg_feat)
    #     if bridge:
    #         return rpn_cls_score, rpn_bbox_pred, rpn_iou_pred, x
    #     return rpn_cls_score, rpn_bbox_pred, rpn_iou_pred

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):

        outs = self(x, self.bridge)
        if self.bridge:
            rpn_cls_score, rpn_bbox_pred, rpn_iou_pred, rpn_feats = outs
            outs = (rpn_cls_score, rpn_bbox_pred, rpn_iou_pred)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            if self.bridge:
                return losses, proposal_list, rpn_feats
            return losses, proposal_list

    def forward(self, feats, bridge=False):
        return multi_apply(self.forward_single, feats, self.scales, [bridge]*len(self.scales))

    def loss_single(self, anchors, cls_score, bbox_pred, iou_pred, labels,
                    label_weights, bbox_targets, num_total_samples):

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_ious = iou_pred[pos_inds]

            if self.reg_decoded_bbox:
                pos_decode_bbox_pred = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_pred)
                pos_encode_bbox_targets = self.bbox_coder.encode(
                    pos_anchors, pos_bbox_targets)
                iou_target = bbox_overlaps(
                    pos_decode_bbox_pred.detach(), pos_bbox_targets, is_aligned=True)
                if self.with_aug_loss:
                    bbox_weights_aug = torch.ones_like(pos_bbox_pred)
                    bbox_weights_aug = (bbox_weights_aug * (iou_target**self.gamma)[:,None])
                    loss_bbox_aug = self.aug_loss(pos_bbox_pred, pos_encode_bbox_targets, bbox_weights_aug.clamp(min=EPS),avg_factor=1.0)

                    # loss_bbox_aug = self.aug_loss(pos_bbox_pred,
                    #                              pos_encode_bbox_targets,
                    #                              reduction_override='none')
                    # bbox_weights_aug_norm = bbox_weights_aug * (loss_bbox_aug.sum()/ (bbox_weights_aug*loss_bbox_aug).sum())
                    # loss_bbox_aug = (loss_bbox_aug * bbox_weights_aug_norm.detach() ).sum()

                bbox_weights = (iou_target**self.gamma)
                # regression loss
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_bbox_targets,
                    weight=bbox_weights.clamp(min=EPS),
                    avg_factor=1.0)

                # loss_bbox = self.loss_bbox(
                #     pos_decode_bbox_pred,
                #     pos_bbox_targets,
                #     reduction_override='none')
                #
                # bbox_weights_norm = bbox_weights * (loss_bbox.sum()/ (bbox_weights*loss_bbox).sum())
                # loss_bbox = (loss_bbox * bbox_weights_norm.detach()).sum()



                if self.with_aug_loss:
                    loss_bbox = (loss_bbox + loss_bbox_aug) * 0.5

            else:
                pos_decode_bbox_pred = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_pred)
                pos_decode_bbox_targets = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_targets)
                iou_target = bbox_overlaps(
                    pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
                bbox_weights = torch.ones_like(pos_bbox_pred)
                bbox_weights = bbox_weights * (iou_target**self.gamma)[:,None]
                loss_bbox = self.loss_bbox(
                    pos_bbox_pred,
                    pos_bbox_targets,
                    bbox_weights.clamp(min=EPS),
                    avg_factor=1.0)
            if isinstance(self.loss_bbox, GHMR):
                avg_factor = iou_target.new_tensor(1.0)
            else:
                # avg_factor = (iou_target**self.gamma).sum()
                avg_factor = (iou_target).sum()
                # avg_factor = iou_target.new_tensor(num_total_samples)

            # centerness loss
            loss_iou = self.loss_centerness(
                pos_ious,
                iou_target,
                avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
            iou_target = bbox_targets.new_tensor(0.)
            avg_factor = iou_target.sum()
        # classification loss
        if isinstance(self.loss_cls, VarifocalLoss):
            cls_iou_targets = torch.zeros_like(cls_score)
            cls_iou_targets[pos_inds] = iou_target.unsqueeze(-1)
            loss_cls = self.loss_cls(
                cls_score, cls_iou_targets, avg_factor=num_total_samples)
        else:
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_iou, avg_factor

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_weights_list, bbox_targets_list, bboxes_weight, pos_inds,
         pos_gt_index, anchor_list, valid_flag_list) = cls_reg_targets

        labels_list = images_to_levels(labels_list, num_level_anchors)
        label_weights_list = images_to_levels(label_weights_list, num_level_anchors)
        anchor_list = images_to_levels(anchor_list, num_level_anchors)
        bbox_targets_list = images_to_levels(bbox_targets_list, num_level_anchors)

        num_total_pos=sum([pos_ind.numel() for pos_ind in pos_inds])
        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_iou,\
            bbox_avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                iou_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_bbox=losses_bbox,
            loss_rpn_iou=losses_iou)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert with_nms, '``with_nms`` in RPNHead should always True'
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            iou_pred_list = [
                iou_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, iou_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        if self.bridge:
            return (result_list, cls_scores, bbox_preds, iou_preds)
        return result_list

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
    ):


        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))


        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            num_level_anchors_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)


        if self.atss:
            (anchors, labels, label_weights, bbox_targets, bbox_weights,
             pos_inds, neg_inds) = results
            gt_inds = None
        else:
            (labels, label_weights, bbox_targets, bbox_weights, valid_pos_inds,
             valid_neg_inds, sampling_result) = results
            # Due to valid flag of anchors, we have to calculate the real pos_inds
            # in origin anchor set.
            pos_inds = []
            for i, single_labels in enumerate(labels):
                pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
                pos_inds.append(pos_mask.nonzero().view(-1))

            gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                gt_inds, concat_anchor_list, concat_valid_flag_list)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            num_level_anchors_list,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        This method is same as `AnchorHead._get_targets_single()`.
        """
        assert unmap_outputs, 'We must map outputs back to the original' \
            'set of anchors in PAAhead'
        if self.atss:
            return self._get_target_single_atss(
                flat_anchors,
                valid_flags,
                num_level_anchors_list,
                gt_bboxes,
                gt_bboxes_ignore,
                gt_labels,
                img_meta,
                label_channels=1,
                unmap_outputs=True
            )
        return super(RPNHead, self)._get_targets_single(
            flat_anchors,
            valid_flags,
            gt_bboxes,
            gt_bboxes_ignore,
            gt_labels,
            img_meta,
            label_channels=1,
            unmap_outputs=True)

    def _get_target_single_atss(self,
                       flat_anchors,
                       valid_flags,
                       num_level_anchors,
                       gt_bboxes,
                       gt_bboxes_ignore,
                       gt_labels,
                       img_meta,
                       label_channels=1,
                       unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # if hasattr(self, 'bbox_coder'):
            #     pos_bbox_targets = self.bbox_coder.encode(
            #         sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            # else:
            #     # used in VFNetHead
            #     pos_bbox_targets = sampling_result.pos_gt_bboxes

            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_iou_pred = iou_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            rpn_iou_pred = rpn_iou_pred.permute(1, 2, 0).reshape(-1).sigmoid()
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            scores = (scores*rpn_iou_pred).sqrt()
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        if proposals.numel() > 0:
            dets, keep = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

    def simple_test_rpn(self, x, img_metas, bridge=False):
        if bridge:
            cls_scores, bbox_preds, iou_preds, x = self(x, bridge=bridge)
        else:
            cls_scores, bbox_preds, iou_preds = self(x)
        rpn_outs = (cls_scores, bbox_preds, iou_preds)
        if self.bridge:
            rpn_results = self.get_bboxes(*rpn_outs, img_metas)
            proposal_list, cls_scores, bbox_preds, iou_preds = rpn_results
        else:
            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        if bridge:
            return rpn_results, x
        else:
            return proposal_list
