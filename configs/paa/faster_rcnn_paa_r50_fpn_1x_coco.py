_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    rpn_head=dict(
        type='PAARPNHead',
        reg_decoded_bbox=True,
        score_voting=True,
        topk=9,
        in_channels=256,
        feat_channels=256,
        # anchor_generator=dict(
        #     _delete_=True,
        #     type='AnchorGenerator',
        #     octave_base_scale=4,
        #     scales_per_octave=3,
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[8, 16, 32, 64, 128]),
        anchor_generator=dict(
            _delete_=True,
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            # strides=[4, 8, 16, 32, 64]),
            strides=[8, 16, 32, 64, 128]),
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     scales=[8],
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
            # target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_centerness =dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.3)),
    roi_head=dict(
        # type='StandardRoIHead',
        type='ProbRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        bbox_head=dict(
            type='ProbShared2FCBBoxHead',
            # type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            # loss_cls=dict(
            #         type='FocalLoss',
            #         use_sigmoid=True,
            #         gamma=1.0,
            #         alpha=0.5,
            #         loss_weight=2.0),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            sampler=None,
        ),
        rpn_proposal = dict(
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=256,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100)
    )
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
