dataset_type = 'SysuDataset'
data_root = 'data/sysu/'
# dataset_type = 'PrwDataset'
# data_root = 'data/prw/'
# dataset_type = 'MotDataset'
# data_root = 'data/mot/'
img_size = (1333, 800)  # (1333, 800), (1500, 900)
batch_size = 9
with_reid = True
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
query_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_bboxes']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
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
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=[
    dict(
        with_reid=with_reid,
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    dict(
        with_reid=with_reid,
        type='MotDataset',
        ann_file='data/mot/' + 'annotations/train.json',
        img_prefix='data/mot/' + 'images/',
        pipeline=train_pipeline),
    ],
    query=dict(
        with_reid=with_reid,
        type=dataset_type,
        ann_file=data_root + 'annotations/query.json',
        img_prefix=data_root + 'images/',
        pipeline=query_pipeline,
        is_query=True),
    test=dict(
        with_reid=with_reid,
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline,
        is_test=True))
evaluation = dict(interval=1, metric='bbox')
