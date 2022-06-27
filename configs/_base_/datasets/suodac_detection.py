dataset_type = 'VOCDataset'
classes = ('echinus','starfish','holothurian','scallop')
data_root = 'data/S-UODAC2020/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromSUODAC', to_float32=True, train=True, domain_file=data_root+'VOC2007/ImageSets'),
    # dict(type='GeneratePuzzle',img_norm_cfg=img_norm_cfg,jig_classes = 30),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'img_puzzle', 'jig_labels']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'domain_label']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromSUODAC', to_float32=True, train=True, domain_file=data_root+'VOC2007/ImageSets'),
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 16,
    train = (
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type1.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type2.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type3.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type4.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type5.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                ann_file=data_root + 'VOC2007/ImageSets/type6.txt',
                img_prefix=data_root + "VOC2007/",
                pipeline=train_pipeline),
    ),
    val=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            ann_file=data_root + 'VOC2007/ImageSets/type7.txt',
            img_prefix=data_root + "VOC2007/",
            pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
