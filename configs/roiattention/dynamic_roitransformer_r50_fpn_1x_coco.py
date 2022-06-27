_base_ = '../faster_rcnn/roitransformer_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='Dynamic_TrHead',
        num_token=20,
        inchannel=7 * 7 * 256,
        emb_dim=256,
        num_heads=4,
        mlp_dim=1024,
        depth=2,
        mode='external_attention',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))
    ))
train_cfg = dict(
    rpn_proposal=dict(nms_thr=0.85),
    rcnn=dict(
        dynamic_rcnn=dict(
            iou_topk=75,
            beta_topk=10,
            update_iter_interval=100,
            initial_iou=0.4,
            initial_beta=1.0)))
test_cfg = dict(rpn=dict(nms_thr=0.85))