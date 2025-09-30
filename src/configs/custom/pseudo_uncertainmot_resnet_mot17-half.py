_base_ = [
    '../_base_/models/yolox_x.py',
    '../_base_/datasets/mot_challenge.py', '../_base_/default_runtime.py'
]

# some hyper parameters
img_scale = (800, 1440)
samples_per_gpu = 1
num_gpus = 3
total_epochs = 3
num_last_epochs = 1
resume_from = None
interval = 5
exp_name = "pseudo_uncertainmot_yolox_x_3x6_mot17-half"
out_dir = "/home/results/" + exp_name

model = dict(
    type='PseudoUncertainMOT',
    detector=dict(
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(
            num_classes=1,
            # post_process="covariance_intersection",
            loss_l1=dict(type='ESLoss', 
                         loss_type="L1")
        ),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            # f"/home/results/{weights_path}/latest.pth"  # noqa: E501
            '/home/allynbao/project/UncertaintyTrack/src/work_dirs/test_run/latest.pth'
        )
    ),
    motion=dict(type='KalmanFilterWithUncertainty'),
    tracker=dict(
        type='ProbabilisticByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30,
        with_covariance=True,
        det_score_mode='confidence',
        final_matching=True,
        expand_boxes=True,
        init_percent=0.7,
        final_percent=0.3,
        init_ellipse_filter=True,
        second_ellipse_filter=True,
        final_ellipse_filter=True,
        use_mahalanobis=False,
        return_expanded=False,
        return_remaining_expanded=False,
        # primary_cascade=dict(num_bins=None),
        primary_cascade=None,
        # secondary_fn=dict(type="wasserstein", threshold=7.5e3),
        secondary_fn=None,
        secondary_cascade=dict(num_bins=None)))
        # secondary_cascade=None))

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
evaluation = dict(metric=['bbox', 'track'])

#*----------------------------------------------------------

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=img_scale,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=[
                # '/home/data/MOT17/annotations/half-train_cocoformat.json',
                # '/home/data/crowdhuman/annotations/crowdhuman_train.json',
                # '/home/data/crowdhuman/annotations/crowdhuman_val.json'
                '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-train_cocoformat.json',
                '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-val_cocoformat.json'
            ],
            img_prefix=[
                # '/home/data/MOT17/train', 
                # '/home/data/crowdhuman/train',
                # '/home/data/crowdhuman/val'
                '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train'
                '/home/allynbao/project/UncertaintyTrack/src/data/MOT17/test'
            ],
            classes=('pedestrian', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20),
        ref_img_sampler=None,
        ))
    # test=dict(
    #     type='MOTChallengeDataset',  # or 'CocoVideoDataset' if needed by your tracker
    #     ann_file='/home/allynbao/project/UncertaintyTrack/src/data/MOT17/annotations/half-train_cocoformat.json',
    #     img_prefix='/home/allynbao/project/UncertaintyTrack/src/data/MOT17/train',
    #     classes=('pedestrian', ),
    #     pipeline=test_pipeline
    # ))

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))

checkpoint_config = dict(interval=2, max_keep_ckpts=2)
evaluation = dict(metric=['bbox', 'track'], interval=interval)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

optimizer_config = dict(grad_clip=None)