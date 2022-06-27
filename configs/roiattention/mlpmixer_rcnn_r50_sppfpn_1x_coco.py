_base_ = 'mlpmixer_rcnn_r50_fpn_1x_coco.py'

model = dict(
    neck=dict(
        type='SPPFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        SPP_type='ASPP_share'),
)
