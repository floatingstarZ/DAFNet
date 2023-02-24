dataset_type = 'CocoDataset'
data_root = './data/jinan/jinan_coco/pan/'


building_module = dict(
    num_classes=1,
    max_per_img=300,
    iou_thr=0.3
)
img_norm_cfg = dict(
    mean=[123.675, 123.675, 116.28, 103.53, 123.675],
    std=[58.395, 58.395, 57.12, 57.375, 58.395], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiSourceTifFromFile',
         fusion_type='pmC',
         src_types=['pan', 'ms', 'fusion', 'blur_pan']),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMultiSourceTifFromFile',
         fusion_type='pmC',
         src_types=['pan', 'ms', 'fusion', 'blur_pan']),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ['building']
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/JPEGImages',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/JPEGImages',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val/val.json',
        img_prefix=data_root + 'val/JPEGImages',
        classes=classes,
        pipeline=test_pipeline))
# evaluation = dict(interval=1, metric=['bbox'])
evaluation = dict(interval=1,
                  metric=['bbox'],
                  metric_items=[
                      'mAP', 'mAP_50', 'mAP_75',
                      'mAP_s', 'mAP_m', 'mAP_l',
                      'AR@100', 'AR@300',
                      'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'
                  ])
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[8, 11])
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]