_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/brackish_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=6))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu = 4,
    workers_per_gpu = 8)