import numpy as np
import torch
from ..registry import PIPELINES

gain_ratio = {'Gain0Gain0': 80.0, 'Gain15Gain0': 15.0, 'Gain30Gain0': 2.5}

def rescale(lq, gt, ratio, lq_format='raw', gt_format='rgb'):
    if lq_format == 'raw':
        lq = np.float32(lq / 65535.0) * ratio
    elif lq_format == 'rgb':
        lq = np.float32(lq / 255.0) * 1.0
    else:
        raise ValueError('lq_format should be raw or rgb')

    lq = np.clip(lq, 0.0, 1.0)

    if gt_format == 'raw':
        gt = np.float32(gt / 65535.0)
    elif gt_format == 'rgb':
        gt = np.float32(gt / 255.0)
    else:
        raise ValueError('gt_format should be raw or rgb')

    return lq, gt

def to_tensor(lq, gt):
    # [T, H, W, C] -> [T, C, H, W]
    lq = torch.from_numpy(lq.transpose(0, 3, 1, 2))
    gt = torch.from_numpy(gt.transpose(0, 3, 1, 2))

    return lq, gt

@PIPELINES.register_module()
class SMOIDnpyTrainPipeline:
    def __init__(self, crop_size, lq_format='raw', gt_format='rgb', flip_ratio=[0.5, 0.5, 0.5], transpose_ratio=0.5, rescale_ratio=1.0):
        self.crop_size = crop_size
        self.lq_format = lq_format
        self.gt_format = gt_format
        self.flip_ratio = flip_ratio
        self.transpose_ratio = transpose_ratio
        self.rescale_ratio = rescale_ratio

    def crop(self, lq, gt, num_input_frames, sequence_length):
        # lq.shape = [T, H, W, 4]
        # gt.shape = [T, 2*H, 2*W, 3]
        data_h, data_w = lq.shape[1], lq.shape[2]
        crop_h, crop_w = self.crop_size

        start_frame = np.random.randint(0, sequence_length - num_input_frames + 1)
        h_offset = np.random.randint(0, data_h - crop_h + 1)
        w_offset = np.random.randint(0, data_w - crop_w + 1)

        if self.lq_format == 'raw' and self.gt_format == 'rgb':
            k = 2
        else:
            k = 1

        cropped_lq = lq[start_frame:start_frame+num_input_frames, h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]
        cropped_gt = gt[start_frame:start_frame+num_input_frames, k*h_offset:k*(h_offset+crop_h), k*w_offset:k*(w_offset+crop_w)]

        return cropped_lq, cropped_gt, start_frame
    
    def flip(self, lq, gt):
        for i in range(3):
            if np.random.random() < self.flip_ratio[i]:
                lq = np.flip(lq, axis=i)
                gt = np.flip(gt, axis=i)
                
        return lq, gt
    
    def transposeHW(self, lq, gt):
        if np.random.random() < self.transpose_ratio:
            lq = np.transpose(lq, (0, 2, 1, 3))
            gt = np.transpose(gt, (0, 2, 1, 3))
        
        return lq, gt
    
    def __call__(self, results):
        cropped_lq, cropped_gt, start_frame = self.crop(results['lq'], results['gt'], 
                                results['num_input_frames'], results['sequence_length'])
        
        gain = results['key'].split('/')[1]
        rescale_ratio = gain_ratio[gain]
        
        lq, gt = self.flip(cropped_lq, cropped_gt)
        lq, gt = self.transposeHW(lq, gt)
        lq, gt = rescale(lq, gt, rescale_ratio, self.lq_format, self.gt_format)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=start_frame)


@PIPELINES.register_module()
class SMOIDnpyValPipeline:
    def __init__(self, crop_size, val_num, rescale_ratio=1.0, lq_format='raw', gt_format='rgb'):
        self.rescale_ratio = rescale_ratio
        self.lq_format = lq_format
        self.gt_format = gt_format
        self.crop_size = crop_size
        self.val_num = val_num

    def crop(self, lq, gt, num_input_frames):
        # lq.shape = [T, H, W, 4]
        # gt.shape = [T, 2*H, 2*W, 3]
        data_h, data_w = lq.shape[1], lq.shape[2]
        crop_h, crop_w = self.crop_size

        start_frame = 0
        h_offset = (data_h//2) - (crop_h//2)
        w_offset = (data_w//2) - (crop_w//2)

        if self.lq_format == 'raw' and self.gt_format == 'rgb':
            k = 2
        else:
            k = 1

        cropped_lq = lq[start_frame:start_frame+num_input_frames, h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]
        cropped_gt = gt[start_frame:start_frame+num_input_frames, k*h_offset:k*(h_offset+crop_h), k*w_offset:k*(w_offset+crop_w)]

        return cropped_lq, cropped_gt, start_frame

    def __call__(self, results):
        lq, gt, start_frame = self.crop(results['lq'], results['gt'], self.val_num)
        gain = results['key'].split('/')[1]
        rescale_ratio = gain_ratio[gain]
        lq, gt = rescale(lq, gt, rescale_ratio, self.lq_format, self.gt_format)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=start_frame)


@PIPELINES.register_module()
class SMOIDnpyTestPipeline:
    def __init__(self, rescale_ratio=1.0, lq_format='raw', gt_format='rgb'):
        self.rescale_ratio = rescale_ratio
        self.lq_format = lq_format
        self.gt_format = gt_format

    def __call__(self, results):
        gain = results['key'].split('/')[1]
        rescale_ratio = gain_ratio[gain]
        lq, gt = rescale(results['lq'], results['gt'], rescale_ratio, self.lq_format, self.gt_format)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=0)