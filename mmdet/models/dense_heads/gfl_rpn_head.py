import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
import copy
from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .gfl_head import Integral
from .atss_rpn_head import ATSSRPNHead
from mmcv.ops import batched_nms
EPS = 1e-12

@HEADS.register_module()
class GFLRPNHead(ATSSRPNHead):
    def __init__(self,
                 *args,
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 reg_max=16,
                 reg_topk=4,
                 add_mean=True,
                 reg_channels=64,
                 **kwargs):
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.total_dim = reg_topk
        self.add_mean = add_mean
        self.reg_channels = reg_channels
        if add_mean:
            self.total_dim += 1
        super(GFLRPNHead, self).__init__(*args, **kwargs)
        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)



    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.rpn_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.rpn_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.rpn_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.rpn_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4 * (self.reg_max + 1), 3, padding=1)
        self.rpn_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

        ##############V2################
        conf_vector = [nn.Conv2d(self.num_anchors * 4 * self.total_dim, self.num_anchors * self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.num_anchors * self.reg_channels, self.num_anchors, 1), nn.Sigmoid()]

        self.reg_conf = nn.Sequential(*conf_vector)
        ##############V2################

    def forward(self, feats, bridge=False):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        for conv in self.rpn_convs:
            x = conv(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = scale(self.rpn_reg(x)).float()
        rpn_iou_pred = self.rpn_iou(x)
        ##############V2################
        N, C, H, W = rpn_bbox_pred.size()
        prob = F.softmax(rpn_bbox_pred.reshape(N, self.num_anchors*4, self.reg_max + 1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
        rpn_cls_score = rpn_cls_score.sigmoid() * quality_score
        ####de-sigmoid####
        rpn_cls_score = torch.log(rpn_cls_score/(1 - rpn_cls_score))

        ##############V2################
        return rpn_cls_score, rpn_bbox_pred, rpn_iou_pred

    def anchor_center(self, anchors):
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, iou_pred, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_ious = iou_pred[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners)

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            iou_target = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            if self.with_aug_loss:
                pos_encode_bbox_targets = self.bbox_coder.encode(
                    pos_anchors, pos_bbox_targets)
                pos_encode_bbox_pred = self.bbox_coder.encode(
                    pos_anchors, pos_decode_bbox_pred*stride[0])
                bbox_weights_aug = torch.ones_like(pos_encode_bbox_pred)
                bbox_weights_aug = bbox_weights_aug * (iou_target ** self.gamma)[:, None]
                loss_box_aug = self.aug_loss(pos_encode_bbox_pred, pos_encode_bbox_targets, bbox_weights_aug.clamp(min=EPS),
                                             avg_factor=1.0)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            bbox_weights = iou_target ** self.gamma
            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=bbox_weights,
                avg_factor=1.0)
            if self.with_aug_loss:
                loss_bbox = (loss_bbox + loss_box_aug) * 0.5

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=bbox_weights[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # loss_dfl = self.loss_dfl(
            #     pred_corners,
            #     target_corners,
            #     weight=iou_target[:, None].expand(-1, 4).reshape(-1),
            #     avg_factor=4.0)

            # centerness loss
            loss_iou = self.loss_centerness(
                pos_ious,
                iou_target,
                avg_factor=num_total_samples)
            avg_factor = (iou_target).sum()

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_iou = iou_pred.sum() * 0
            iou_target = bbox_targets.new_tensor(0.)
            avg_factor = iou_target.sum()
            # weight_targets = bbox_pred.new_tensor(0)

        # cls loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_iou, avg_factor

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
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

        losses_cls, losses_bbox, losses_dfl,losses_iou,\
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                iou_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox, loss_rpn_dfl=losses_dfl,loss_rpn_iou=losses_iou)

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
                                                mlvl_anchors, self.anchor_generator.strides, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           iou_preds,
                           mlvl_anchors,
                           strides,
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
        # mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_iou_pred = iou_preds[idx]
            stride = strides[idx]
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
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0)
            anchors = mlvl_anchors[idx]
            pos_anchor_centers = self.anchor_center(anchors)
            pos_bbox_pred_corners = self.integral(rpn_bbox_pred)*stride[0]
            rpn_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_corners, max_shape=img_shape)
            rpn_bbox_pred = rpn_bbox_pred.reshape(-1, 4)
            scores = (scores*rpn_iou_pred).sqrt()
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                # anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            # mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        # anchors = torch.cat(mlvl_valid_anchors)
        proposals = torch.cat(mlvl_bbox_preds)
        # proposals = self.bbox_coder.decode(
        #     anchors, rpn_bbox_pred, max_shape=img_shape)
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

    # def _get_bboxes(self,
    #                 cls_scores,
    #                 bbox_preds,
    #                 iou_preds,
    #                 mlvl_anchors,
    #                 img_shapes,
    #                 scale_factors,
    #                 cfg,
    #                 rescale=False,
    #                 with_nms=True):
    #
    #     cfg = self.test_cfg if cfg is None else cfg
    #     assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
    #     batch_size = cls_scores[0].shape[0]
    #
    #     mlvl_bboxes = []
    #     mlvl_scores = []
    #     for cls_score, bbox_pred, stride, anchors in zip(
    #             cls_scores, bbox_preds, self.anchor_generator.strides,
    #             mlvl_anchors):
    #         assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
    #         assert stride[0] == stride[1]
    #         scores = cls_score.permute(0, 2, 3, 1).reshape(
    #             batch_size, -1, self.cls_out_channels).sigmoid()
    #         bbox_pred = bbox_pred.permute(0, 2, 3, 1)
    #
    #         bbox_pred = self.integral(bbox_pred) * stride[0]
    #         bbox_pred = bbox_pred.reshape(batch_size, -1, 4)
    #
    #         nms_pre = cfg.get('nms_pre', -1)
    #         if nms_pre > 0 and scores.shape[1] > nms_pre:
    #             max_scores, _ = scores.max(-1)
    #             _, topk_inds = max_scores.topk(nms_pre)
    #             batch_inds = torch.arange(batch_size).view(
    #                 -1, 1).expand_as(topk_inds).long()
    #             anchors = anchors[topk_inds, :]
    #             bbox_pred = bbox_pred[batch_inds, topk_inds, :]
    #             scores = scores[batch_inds, topk_inds, :]
    #         else:
    #             anchors = anchors.expand_as(bbox_pred)
    #
    #         bboxes = distance2bbox(
    #             self.anchor_center(anchors), bbox_pred, max_shape=img_shapes)
    #         mlvl_bboxes.append(bboxes)
    #         mlvl_scores.append(scores)
    #
    #     batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    #     if rescale:
    #         batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
    #             scale_factors).unsqueeze(1)
    #
    #     batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
    #     # Add a dummy background class to the backend when using sigmoid
    #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    #     # BG cat_id: num_class
    #     padding = batch_mlvl_scores.new_zeros(batch_size,
    #                                           batch_mlvl_scores.shape[1], 1)
    #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
    #
    #     if with_nms:
    #         det_results = []
    #         for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
    #                                               batch_mlvl_scores):
    #             det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
    #                                                  cfg.score_thr, cfg.nms,
    #                                                  cfg.max_per_img)
    #             det_results.append(tuple([det_bbox, det_label]))
    #     else:
    #         det_results = [
    #             tuple(mlvl_bs)
    #             for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
    #         ]
    #     return det_results
