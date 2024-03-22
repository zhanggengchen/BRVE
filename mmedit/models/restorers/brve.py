# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os.path as osp

import mmcv
import numpy as np
import cv2
import torch

from mmedit.core import tensor2img, tensor2numpy
from ..registry import MODELS
from .basic_restorer import BasicRestorer
from mmedit.utils import simple_isp, LLRVD_isp
from mmedit.core.evaluation import strred

import torch.nn.functional as F


@MODELS.register_module()
class BRVE(BasicRestorer):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        ensemble (dict): Config for ensemble. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 ensemble=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 clip_length=150):
        super().__init__(generator, pixel_loss, train_cfg, test_cfg,
                         pretrained)

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False
        self.clip_length = clip_length

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

        # ensemble
        self.forward_ensemble = None
        if ensemble is not None:
            if ensemble['type'] == 'SpatialTemporalEnsemble':
                from mmedit.models.common.ensemble import \
                    SpatialTemporalEnsemble
                is_temporal = ensemble.get('is_temporal_ensemble', False)
                self.forward_ensemble = SpatialTemporalEnsemble(is_temporal)
            else:
                raise NotImplementedError(
                    'Currently support only '
                    '"SpatialTemporalEnsemble", but got type '
                    f'[{ensemble["type"]}]')

    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """

        is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                is_mirror_extended = True

        return is_mirror_extended

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        # fix SPyNet and EDVR at the beginning
        if (self.step_counter < self.fix_iter) or (self.fix_iter == -1):
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if ('spynet' in k) or ('edvr' in k) or ('pwcnet' in k):
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        self.step_counter += 1

        outputs.update({'log_vars': log_vars})
        return outputs

    def evaluate(self, output, gt, save_path=None):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        gt_format = 'raw'
        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                for i in range(0, output.size(1)):
                    output_i = tensor2numpy(output[:, i, :, :, :])
                    gt_i = tensor2numpy(gt[:, i, :, :, :])
                    avg.append(self.allowed_metrics[metric](output_i, gt_i, gt_format))
                eval_result[metric] = np.mean(avg)

        # STRRED
        assert output.shape[0] == 1
        assert gt.shape[0] == 1
        output_npy = np.transpose(output.squeeze(0).detach().cpu().numpy(), (0, 2, 3, 1))
        gt_npy = np.transpose(gt.squeeze(0).detach().cpu().numpy(), (0, 2, 3, 1))
        eval_result['strred'] = strred(gt_npy, output_npy)

        return eval_result
    
    def evaluate_rgb(self, output, gt, meta):
        """Evaluation function.

        If the output contains multiple frames, we compute the metric
        one by one and take an average.

        Args:
            output (Tensor): Model output with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        gt_format = 'rgb'
        folder_name = meta[0]['key'].split('/')[0]
        ISP_info_path = self.test_cfg.get('isp_info_path', "datasets/LLRVD/ISP_info")

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if output.ndim == 5:  # a sequence: (n, t, c, h, w)
                avg = []
                output_rgb = []
                gt_rgb = []
                for i in range(0, output.size(1)):
                    output_i = tensor2numpy(output[:, i, :, :, :])
                    output_i = LLRVD_isp(output_i, folder_name, i+1, ISP_info_path)
                    output_rgb.append(output_i)

                    gt_i = tensor2numpy(gt[:, i, :, :, :])
                    gt_i = LLRVD_isp(gt_i, folder_name, i+1, ISP_info_path)
                    gt_rgb.append(gt_i)

                    avg.append(self.allowed_metrics[metric](output_i, gt_i, gt_format))
                eval_result[metric] = np.mean(avg)

        output_rgb = np.stack(output_rgb, axis=0)
        gt_rgb = np.stack(gt_rgb, axis=0)
        # STRRED
        eval_result['strred'] = strred(gt_rgb, output_rgb, gt_format)

        return eval_result


    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, t, c, h, w).
            gt (Tensor): GT Tensor with shape (n, t, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        with torch.no_grad():
            b, t, c, h, w = lq.shape
            output = torch.zeros((b, t, 4, h, w))
            for i in range(0, t, self.clip_length):
                print(i, i+self.clip_length)
                if self.test_cfg is not None and self.test_cfg.get('pad', None):
                    pad = self.test_cfg['pad']
                    output[:, i:i+self.clip_length] = self.pad2D_test(lq[:, i:i+self.clip_length], pad)
                # output[:, i:i+self.clip_length] = self.generator(lq[:, i:i+self.clip_length])[:, :, :, :h, :w]
                elif self.test_cfg is not None and self.test_cfg.get('tile', None):
                    tile = self.test_cfg['tile']
                    tile_overlap = self.test_cfg['tile_overlap']
                    output[:, i:i+self.clip_length] = self.tile2D_test(lq[:, i:i+self.clip_length], tile, tile_overlap)
                else:
                    output[:, i:i+self.clip_length] = self.generator(lq[:, i:i+self.clip_length])
            output = output[:, :t, :, :, :]
            print(output.shape)

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        if gt is not None and gt.ndim == 4:
            t = output.size(1)
            if self.check_if_mirror_extended(lq):  # with mirror extension
                output = 0.5 * (output[:, t // 4] + output[:, -1 - t // 4])
            else:  # without mirror extension
                output = output[:, t // 2]

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            if self.test_cfg.get('test_format', 'raw') == 'raw':
                results = dict(scene=meta[0]['key'].split('/')[0], gain=meta[0]['key'].split('/')[1], eval_result=self.evaluate(output, gt))
            else:
                results = dict(scene=meta[0]['key'].split('/')[0], gain=meta[0]['key'].split('/')[1], eval_result=self.evaluate_rgb(output, gt, meta))
            print(results)
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            self.save_img(output, meta, save_path, iteration)

        return results
    
    def save_img(self, output, meta, save_path, iteration):
        # 网络输出的格式
        out_format = self.test_cfg.get('gt_format', 'rgb')
        used_isp = self.test_cfg.get('used_isp', 'simple_isp')
        ISP_info_path = self.test_cfg.get('isp_info_path', "datasets/LLRVD/ISP_info")
        folder_name = meta[0]['key'].split('/')[0]
        gain = meta[0]['key'].split('/')[1]
        
        save_path_v = osp.join(save_path, f'{folder_name}_{gain}.mp4')
        
        video_resolution = self.test_cfg.get('video_resolution', (640, 480))
        frame_freq = self.test_cfg.get('frame_freq', 15)
        video = cv2.VideoWriter(save_path_v, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_freq, video_resolution, True)
        print(save_path_v) # save_path_v 需要存在
        # print(flow_save_path_v)

        for i in range(0, output.size(1)):
            if isinstance(iteration, numbers.Number):
                save_path_i = osp.join(
                    save_path, f"{folder_name}_{gain}",
                    f'{i:08d}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path_i = osp.join(save_path, f"{folder_name}_{gain}",
                                        f'{i:08d}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                    f'but got {type(iteration)}')
            if out_format == 'raw':
                if used_isp == 'simple_isp':
                    frame_i = tensor2img(output[:, i, :, :, :], out_type=np.uint16, change_order=False)
                    frame_i = simple_isp(frame_i)
                elif used_isp == 'LLRVD_isp':
                    frame_i = tensor2numpy(output[:, i, :, :, :])
                    frame_i = LLRVD_isp(frame_i, folder_name, i+1, ISP_info_path)
            else:
                frame_i = tensor2img(output[:, i, :, :, :], change_order=False)
            mmcv.imwrite(frame_i, save_path_i)
            video.write(frame_i)
        
        video.release()

    def pad2D_test(self, lq, pad):
        mod_pad_h, mod_pad_w = 0, 0
        b, t, c, h, w = lq.size()
        if h % pad != 0:
            mod_pad_h = pad - h % pad
        if w % pad != 0:
            mod_pad_w = pad - w % pad
        img = F.pad(lq, (0, mod_pad_w, 0, mod_pad_h), 'constant')
        print(img.size())
        output = self.generator(img)
        print(output.shape)
        print(output[:, :, :, 0:h, 0:w].shape)
        return output[:, :, :, 0:h, 0:w]
    
    def tile2D_test(self, lq, tile, tile_overlap):
        # test the image tile by tile
        b, t, c, h, w = lq.shape
        tile = min(tile, h, w)
        # assert tile % 8 == 0, "tile size should be multiple of 8"
        tile_overlap = tile_overlap

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, t, 4, h, w).type_as(lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = self.generator(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        restored = E.div_(W)

        print(restored.shape)
        
        return restored

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        with torch.no_grad():
            t = img.shape[1]
            for i in range(0, t, self.clip_length):
                print(i, i+self.clip_length)
                out = self.generator(img[:, i:i+self.clip_length])
        return out