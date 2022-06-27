_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/utdac_detection_coco.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='CascadePAAHead',
        num_classes=4,
        num_stages=2,
        stage_loss_weights=[1, 0.5, 0.25],
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
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
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
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
    ],
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 8)