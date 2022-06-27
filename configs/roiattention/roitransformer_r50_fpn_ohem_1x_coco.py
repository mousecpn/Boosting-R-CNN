_base_ = './roitransformer_r50_fpn_1x_coco.py'
train_cfg = dict(rcnn=dict(sampler=dict(type='OHEMSamplerDh')))
