_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(train_cfg=dict(
    rpn=dict(
        assigner=dict(
            _delete_=True,
            type='CenterRegionAssigner',
            pos_scale=0.2,
            neg_scale=0.2,
            min_pos_iof=0.01),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    # rcnn=dict(
    #     assigner=dict(
    #         type='MaxIoUAssigner',
    #         pos_iou_thr=0.1,
    #         neg_iou_thr=0.1,
    #         min_pos_iou=0,
    #         ignore_iof_thr=-1),
    #     sampler=dict(type='PAASampler',
    #          add_gt_as_proposals=True,
    #          topk=128,
    #          score_voting=True,
    #          covariance_type='diag')
    # )
))
