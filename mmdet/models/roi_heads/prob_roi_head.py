import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .cascade_roi_head import CascadeRoIHead
import numpy as np

@HEADS.register_module()
class ProbRoIHead(StandardRoIHead):
    def __init__(self, alpha=0, gamma=0.1, boost=False, prob=True, ams=False, quality=False, iou_gamma=0, reg_norm='bbox_num', **kwargs):
        super(ProbRoIHead, self).__init__(**kwargs)
        self.alpha  = alpha
        self.gamma = gamma
        self.boost = boost
        self.prob = prob
        self.ams = ams
        self.quality = quality
        self.iou_gamma = iou_gamma
        self.reg_norm = reg_norm

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            priors = []
            ious = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                # prior extraction
                num_gts = assign_result.num_gts
                pos_inds = sampling_result.pos_inds[num_gts:].clone() - num_gts
                neg_inds = sampling_result.neg_inds.clone() - num_gts
                pos_prior = proposal_list[i][pos_inds, -1].clone()
                neg_prior = 1 - proposal_list[i][neg_inds, -1].clone()
                gt_weights = pos_prior.new_zeros(num_gts)
                if self.quality:
                    pos_ious = assign_result.max_overlaps[sampling_result.pos_inds]
                    neg_ious = 1 - assign_result.max_overlaps[sampling_result.neg_inds]
                    iou = torch.cat([pos_ious, neg_ious], dim=0).detach()
                    ious.append(iou)
                prior = torch.cat([gt_weights, pos_prior, neg_prior], dim=0).detach()
                priors.append(prior)
        priors = torch.cat(priors, dim=0)
        if not self.quality:
            ious = None
        else:
            ious = torch.cat(ious, dim=0)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            if self.boost:
                bbox_results = self._bbox_forward_train_boost(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, priors, ious)
            else:
                bbox_results = self._bbox_forward_train(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        if self.ams:
            cls_score, bbox_pred, bbox_feats = self.bbox_head(bbox_feats)
        else:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train_boost(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, priors, ious=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        labels, label_weights, bbox, bbox_weights = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        cls_score = bbox_results['cls_score'].clone().softmax(1).detach()
        cls_score = torch.gather(cls_score, 1, labels.reshape(-1,1))
        if ious is not None:
            if self.alpha == 0:
                # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma
                label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma
            else:
                # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma * self.alpha
                label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma * self.alpha
        else:
            if self.alpha == 0:
                label_weights_new = (1 - priors) ** self.gamma
            else:
                label_weights_new = (1 - priors) ** self.gamma * self.alpha
        # bbox_targets = (labels, label_weights_new, bbox, bbox_weights)
        bbox_targets = (labels, label_weights, bbox, bbox_weights)
        # if self.ams:
        #     loss_bbox = self.bbox_head.loss(bbox_results['bbox_feats'],
        #                                     bbox_results['cls_score'],
        #                                     bbox_results['bbox_pred'], rois,
        #                                     *bbox_targets)
        # else:
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets, reduction_override='none')
        # cls_weights = label_weights_new * (loss_bbox['loss_cls'].sum() / (label_weights_new * loss_bbox['loss_cls']).sum())
        # loss_bbox['loss_cls'] = (loss_bbox['loss_cls']*cls_weights.detach()).mean()

        loss_bbox['loss_cls'] = self.norm_loss(loss_bbox['loss_cls'], label_weights_new, label_weights_new.shape[0])
        if self.reg_norm == 'mean':
            loss_bbox['loss_bbox'] = loss_bbox['loss_bbox'].mean()
        else:
            loss_bbox['loss_bbox'] = loss_bbox['loss_bbox'].sum()/bbox.size(0)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def norm_loss(self,loss,weights,avg_factor):
        new_weights = weights * (loss.sum() / (weights * loss).sum())
        loss = (loss*new_weights.detach()).sum() / avg_factor
        return loss

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.
        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]


    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):

        rois = bbox2roi(proposals)
        prior = torch.cat([boxes[:,-1] for boxes in proposals],dim=0)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        if self.prob:
            cls_score = bbox_results['cls_score'].softmax(1)
            cls_score = cls_score * prior.reshape(-1,1)
            # num_classes = cls_score.shape[1] - 1
            # cls_score[:,:-1] = (cls_score[:,:-1] + prior.reshape(-1,1)/num_classes)/2
            # cls_score[:, -1] = (cls_score[:,-1] + (1 - prior))/2
            # cls_score[:,:-1] = cls_score[:,:-1] * prior.reshape(-1,1)
            # cls_score[:, -1] = cls_score[:,-1] * (1 - prior)
            cls_score = cls_score**0.5
        else:
            cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))
            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

@HEADS.register_module()
class BoostRoIHead(ProbRoIHead):
    def __init__(self, **kwargs):
        super(BoostRoIHead, self).__init__(**kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            priors = []
            ious = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

                # prior extraction
                num_gts = assign_result.num_gts
                pos_inds = sampling_result.pos_inds[num_gts:].clone() - num_gts
                neg_inds = sampling_result.neg_inds.clone() - num_gts
                prior = torch.cat((proposal_list[i][pos_inds, 4:],proposal_list[i][neg_inds, 4:]),dim=0).clone()
                neg_prior = prior.new_zeros(prior.shape[0], 1)
                prior = torch.cat((prior, neg_prior), dim=1)
                neg_prior,_ = proposal_list[i][neg_inds, 4:].clone().max(-1)
                prior[pos_inds.shape[0]:,-1] = neg_prior
                gt_weights = prior.new_zeros(num_gts, prior.shape[1])
                if self.quality:
                    pos_ious = assign_result.max_overlaps[sampling_result.pos_inds]
                    neg_ious = 1 - assign_result.max_overlaps[sampling_result.neg_inds]
                    iou = torch.cat([pos_ious, neg_ious], dim=0).detach()
                    ious.append(iou)
                prior = torch.cat([gt_weights, prior], dim=0).detach()
                priors.append(prior)
        priors = torch.cat(priors, dim=0)
        if not self.quality:
            ious = None
        else:
            ious = torch.cat(ious, dim=0)
        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            if self.boost:
                bbox_results = self._bbox_forward_train_boost(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas, priors, ious)
            else:
                bbox_results = self._bbox_forward_train(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):

        rois = bbox2roi(proposals)
        prior = torch.cat([boxes[:,4:] for boxes in proposals],dim=0)
        neg_prior = prior.new_ones(prior.shape[0], 1)

        prior = torch.cat((prior,neg_prior),dim=1)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        if self.prob:
            cls_score = bbox_results['cls_score'].softmax(1)
            cls_score = cls_score * prior
            cls_score = cls_score**0.5
        else:
            cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))
            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def _bbox_forward_train_boost(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, priors, ious=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        labels, label_weights, bbox, bbox_weights = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        priors = torch.gather(priors, 1, labels.reshape(-1, 1)).squeeze()
        cls_score = bbox_results['cls_score'].clone().softmax(1).detach()
        cls_score = torch.gather(cls_score, 1, labels.reshape(-1,1))
        if ious is not None:
            if self.alpha == 0:
                # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma
                label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma
            else:
                # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma * self.alpha
                label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma * self.alpha
        else:
            if self.alpha == 0:
                label_weights_new = (1 - priors) ** self.gamma
            else:
                label_weights_new = (1 - priors) ** self.gamma * self.alpha
        bbox_targets = (labels, label_weights_new, bbox, bbox_weights)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

from mmdet.models.losses import SmoothL1Loss
EPS = 1e-15

@HEADS.register_module()
class DyProbRoIHead(ProbRoIHead):
    def __init__(self, **kwargs):
        super(DyProbRoIHead, self).__init__(**kwargs)
        assert isinstance(self.bbox_head.loss_bbox, SmoothL1Loss)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history = []

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            priors = []
            cur_iou = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                # record the `iou_topk`-th largest IoU in an image
                iou_topk = min(self.train_cfg.dynamic_rcnn.iou_topk,
                               len(assign_result.max_overlaps))
                ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                cur_iou.append(ious[-1].item())
                sampling_results.append(sampling_result)
                # prior extraction
                num_gts = assign_result.num_gts
                pos_inds = sampling_result.pos_inds[num_gts:].clone() - num_gts
                neg_inds = sampling_result.neg_inds.clone() - num_gts
                pos_prior = proposal_list[i][pos_inds, -1].clone()
                neg_prior = 1 - proposal_list[i][neg_inds, -1].clone()
                gt_weights = neg_prior.new_zeros(num_gts)
                prior = torch.cat([gt_weights, pos_prior, neg_prior], dim=0).detach()
                priors.append(prior)
            priors = torch.cat(priors, dim=0)
            # average the current IoUs over images
            cur_iou = np.mean(cur_iou)
            self.iou_history.append(cur_iou)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            if self.with_bbox:
                if self.boost:
                    bbox_results = self._bbox_forward_train_boost(x, sampling_results,
                                                                  gt_bboxes, gt_labels,
                                                                  img_metas, priors)
                else:
                    bbox_results = self._bbox_forward_train(x, sampling_results,
                                                            gt_bboxes, gt_labels,
                                                            img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        # update IoU threshold and SmoothL1 beta
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            new_iou_thr, new_beta = self.update_hyperparameters()

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        num_imgs = len(img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        pos_inds = bbox_targets[3][:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        cur_target = bbox_targets[2][pos_inds, :2].abs().mean(dim=1)
        beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                        num_pos)
        cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
        self.beta_history.append(cur_target)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train_boost(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, priors):
        """Run forward function and calculate loss for box head in training."""
        num_imgs = len(img_metas)
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)
        labels, label_weights, bbox, bbox_weights = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        pos_inds = bbox_weights[:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        cur_target = bbox[pos_inds, :2].abs().mean(dim=1)
        beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * num_imgs,
                        num_pos)
        cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
        self.beta_history.append(cur_target)
        if self.alpha == 0:
            label_weights_new = (1 - priors) ** self.gamma
        else:
            label_weights_new = (1 - priors) ** self.gamma * self.alpha
        bbox_targets = (labels, label_weights_new, bbox, bbox_weights)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def update_hyperparameters(self):

        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou,
                          np.mean(self.iou_history))
        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr
        if (np.median(self.beta_history) < EPS):
            # avoid 0 or too small value for new_beta
            new_beta = self.bbox_head.loss_bbox.beta
        else:
            new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta,
                           np.median(self.beta_history))
        self.beta_history = []
        self.bbox_head.loss_bbox.beta = new_beta
        return new_iou_thr, new_beta



@HEADS.register_module()
class ProbCascadeRoIHead(CascadeRoIHead):
    def __init__(self, alpha=0, gamma=0.1, boost=False, **kwargs):
        super(ProbCascadeRoIHead, self).__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.boost = boost

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            priors = []
            ious = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
                # prior extraction
                num_gts = assign_result.num_gts
                pos_inds = sampling_result.pos_inds[num_gts:].clone() - num_gts
                neg_inds = sampling_result.neg_inds.clone() - num_gts
                pos_prior = proposal_list[i][pos_inds, -1].clone()
                neg_prior = 1 - proposal_list[i][neg_inds, -1].clone()
                gt_weights = pos_prior.new_zeros(num_gts)
                # if self.quality:
                #     pos_ious = assign_result.max_overlaps[sampling_result.pos_inds]
                #     neg_ious = 1 - assign_result.max_overlaps[sampling_result.neg_inds]
                #     iou = torch.cat([pos_ious, neg_ious], dim=0).detach()
                #     ious.append(iou)
                prior = torch.cat([gt_weights, pos_prior, neg_prior], dim=0).detach()
                priors.append(prior)
            priors = torch.cat(priors, dim=0)

            # bbox head forward and loss
            if self.boost:
                bbox_results = self._bbox_forward_train_boost(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        rcnn_train_cfg, priors)
            else:
                bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses


    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        prior = [boxes[:,-1] for boxes in proposal_list]

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]


        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            cls_score[i] = cls_score[i].softmax(1)
            cls_score[i][:, :-1] = cls_score[i][:, :-1] * prior[i].reshape(-1, 1)
            cls_score[i][:, -1] = cls_score[i][:, -1] * (1 - prior[i])
            cls_score[i] = cls_score[i] ** 0.5
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results


        results = ms_bbox_result['ensemble']

        return results

    def _bbox_forward_train_boost(self, stage, x, sampling_results, gt_bboxes, gt_labels,
                             rcnn_train_cfg, priors):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(stage, x, rois)
        labels, label_weights, bbox, bbox_weights = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        # cls_score = bbox_results['cls_score'].clone().softmax(1).detach()
        # cls_score = torch.gather(cls_score, 1, labels.reshape(-1,1))
        # if ious is not None:
        #     if self.alpha == 0:
        #         # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma
        #         label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma
        #     else:
        #         # label_weights_new = (ious**self.iou_gamma) * (1 - priors) ** self.gamma * self.alpha
        #         label_weights_new = (ious - cls_score).abs()**self.iou_gamma * (1 - priors) ** self.gamma * self.alpha
        # else:
        if self.alpha == 0:
            label_weights_new = (1 - priors) ** self.gamma
        else:
            label_weights_new = (1 - priors) ** self.gamma * self.alpha
        bbox_targets = (labels, label_weights_new, bbox, bbox_weights)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results
