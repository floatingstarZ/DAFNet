_base_ = [
    './base_settings_DAFNet.py',
    './base_faster_rcnn_DAFNet.py'
]

model = dict(
    neck=dict(
        fusion_cfg=dict(
            type='CRGs'
        ),
    ),
)
