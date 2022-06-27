_base_='paa_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        _delete_=True,
        type='CascadePAAHead',
        num_classes=4,
        num_stages=2,
        fusion=False,
        stages=[
            dict(
                type='StageCascadeDenseHead',
                num_classes=4,
                in_channels=256,
                stacked_convs=0,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                reg_decoded_bbox=True,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
            # dict(
            #     type='StageCascadePAAHead',
            #     num_classes=4,
            #     in_channels=256,
            #     feat_channels=256,
            #     stacked_convs=0,
            #     anchor_generator=dict(
            #         type='AnchorGenerator',
            #         scales=[8],
            #         ratios=[1.0],
            #         # strides=[4, 8, 16, 32, 64]),
            #         strides=[8, 16, 32, 64, 128]),
            #     adapt_cfg=dict(type='dilation', dilation=3),
            #     bridged_feature=False,
            #     sampling=False,
            #     with_cls=True,
            #     reg_decoded_bbox=True,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=(.0, .0, .0, .0),
            #         target_stds=(0.1, 0.1, 0.5, 0.5)),
            #     loss_cls=dict(
            #         type='FocalLoss',
            #         use_sigmoid=True,
            #         gamma=2.0,
            #         alpha=0.25,
            #         loss_weight=1.0),
            #     loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            #     loss_centerness=dict(
            #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
            dict(
                type='StageCascadePAAHead',
                num_classes=4,
                in_channels=256,
                feat_channels=256,
                stacked_convs=0,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                sampling=False,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
            # dict(
            #     type='StageCascadePAAHead',
            #     num_classes=4,
            #     in_channels=256,
            #     feat_channels=256,
            #     stacked_convs=0,
            #     anchor_generator=dict(
            #         type='AnchorGenerator',
            #         scales=[8],
            #         ratios=[1.0],
            #         # strides=[4, 8, 16, 32, 64]),
            #         strides=[8, 16, 32, 64, 128]),
            #     adapt_cfg=dict(type='offset'),
            #     bridged_feature=False,
            #     sampling=False,
            #     with_cls=True,
            #     reg_decoded_bbox=True,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=(.0, .0, .0, .0),
            #         target_stds=[0.033, 0.033, 0.067, 0.067]),
            #     # loss_cls=dict(
            #     #     type='CrossEntropyLoss', use_sigmoid=True,
            #     #     loss_weight=1.0),
            #     # loss_bbox=dict(type='IoULoss', linear=True, loss_weight=10.0))
            #     loss_cls=dict(
            #         type='FocalLoss',
            #         use_sigmoid=True,
            #         gamma=2.0,
            #         alpha=0.25,
            #         loss_weight=1.0),
            #     loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            #     loss_centerness=dict(
            #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
        ]),
    train_cfg=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.1,
                neg_iou_thr=0.1,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        # dict(
        #     assigner=dict(
        #         type='MaxIoUAssigner',
        #         pos_iou_thr=0.1,
        #         neg_iou_thr=0.1,
        #         min_pos_iou=0,
        #         ignore_iof_thr=-1),
        #     allowed_border=-1,
        #     pos_weight=-1,
        #     debug=False),
    ],
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# load_from = 'work_dirs/cascade_paa_r50_fpn_1x_coco/epoch_1.pth'
runner = dict(type='EpochBasedRunner', max_epochs=14)
