exp_name = 'BRVE_SMOID'

# model settings
model = dict(
    type='BRVE',
    clip_length=50,
    generator=dict(
        type='BNNVD',
        mid_channels=24,
        feat_extract_blocks=3,
        num_unets=1,
        unet_n_feat=[24, 48, 96],
        unet_n_block=[1, 3, 3],
        stage1_n_feat=[24, 48, 48, 48],
        task='Raw2Raw'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

# model training and testing settings
train_cfg = dict(fix_iter=-1)
test_cfg = dict(metrics=['PSNR', 'SSIM'], gt_format='raw')

# dataset settings
train_dataset_type = 'LLSMOIDnpyDataset'
val_dataset_type = 'LLSMOIDnpyDataset'

# rescale_ratio = 80.0, 15.0, 2.5
train_pipeline = [
    dict(
        type='SMOIDnpyTrainPipeline',
        lq_format = 'raw',
        gt_format = 'raw',
        crop_size=(128, 128)),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'start_frame'])
]

val_pipeline = [
    dict(type='SMOIDnpyValPipeline',
         lq_format = 'raw',
         gt_format = 'raw',
         crop_size=(128, 128),
         val_num=10),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'start_frame'])
]

test_pipeline = [
    dict(type='SMOIDnpyTestPipeline',
         lq_format = 'raw',
         gt_format = 'raw'),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'start_frame'])
]

data = dict(
    workers_per_gpu=0,
    train_dataloader=dict(
        samples_per_gpu=1, shuffle=False, drop_last=True, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=1, shuffle=False, persistent_workers=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=1, persistent_workers=False),

    # train
    train=dict(
        type=train_dataset_type,
        lq_folder='datasets/SMOID',
        gt_folder='datasets/SMOID',
        pipeline=train_pipeline,
        gain='all',
        ann_file='data_list/train_list_DarkRaw2BrightRaw.txt',
        num_input_frames=10,
        test_mode=False),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='datasets/SMOID',
        gt_folder='datasets/SMOID',
        pipeline=val_pipeline,
        gain='all',
        ann_file='data_list/val_list_DarkRaw2BrightRaw10.txt',
        num_input_frames=10,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        lq_folder='datasets/SMOID',
        gt_folder='datasets/SMOID',
        pipeline=test_pipeline,
        gain='all',
        ann_file='data_list/test_list_DarkRaw2BrightRaw.txt',
        memorize=False,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99)))

# learning policy
total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[100000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=2000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
