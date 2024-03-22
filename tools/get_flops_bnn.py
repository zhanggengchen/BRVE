# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmcv import Config
from mmedit.utils import get_model_complexity_info_bnn

from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[250, 250],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    elif len(args.shape) in [3, 4]:  # 4 for video inputs (t, c, h, w)
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported '
            f'with {model.__class__.__name__}')


    config_name = osp.splitext(osp.split(args.config)[1])[0]
    print(config_name)
    with open(f'{config_name}.txt', 'w') as f:
        flops, params = get_model_complexity_info_bnn(model, input_shape, ost=f)
        split_line = '=' * 30
        print(f'{split_line}\nInput shape: {input_shape}\n'
            f'Flops: {flops}\nParams: {params}\n{split_line}', file=f)
    
    # flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    if len(input_shape) == 4:
        print('!!!If your network computes N frames in one forward pass, you '
              'may want to divide the FLOPs by N to get the average FLOPs '
              'for each frame.')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=5 python tools/get_flops_bnn.py configs/bnnvdv2_bbcu_seabs_llrvd_lr2e-4.py --shape 10 4 544 960
# CUDA_VISIBLE_DEVICES=0 python tools/get_flops_bnn.py configs/bnnvdv3_bbcu_seabs_llrvd_lr2e-4.py --shape 100 4 128 128
# CUDA_VISIBLE_DEVICES=2 python tools/get_flops_bnn.py configs/bnnvdv2_bbcu_seabsrep_llrvd_lr2e-4.py --shape 100 4 128 128
# CUDA_VISIBLE_DEVICES=5 python tools/get_flops_bnn.py configs/bnnvdv2_bivit_llrvd_lr2e-4.py --shape 100 4 128 128