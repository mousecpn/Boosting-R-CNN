dataset_type = 'CocoDataset'
classes = ('echinus','starfish','holothurian','scallop')
data_root = 'data/S-UODAC2020/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    # dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadImageFromUODAC',source = ['type1','type2','type3','type4','type5','type6'],train=True),
    # dict(type='LoadImageFromUODAC', source=['type7']),
    # dict(type='GeneratePuzzle',img_norm_cfg=img_norm_cfg,jig_classes = 30),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'img_puzzle', 'jig_labels']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'domain_label']),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromUODAC', source = ['type7']),
    # dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadImageFromUODAC',source = ['type1','type2','type3','type4','type5','type6']),
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
    workers_per_gpu = 2,
    # train = dict(
    #     type = 'DomainBalancedDataset',
    #     source = ['type1','type2','type3','type4','type5','type6'],
    #     dataset =dict(
    #         type=dataset_type,
    #         ann_file='data/S-UODAC2020/COCO_Annotations/instances_source.json',
    #         img_prefix=data_root,
    #         pipeline=train_pipeline)
    # ),
    train=dict(
        type=dataset_type,
        ann_file='data/S-UODAC2020/COCO_Annotations/instances_source.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/S-UODAC2020/COCO_Annotations/instances_target.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/S-UODAC2020/COCO_Annotations/instances_target.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
