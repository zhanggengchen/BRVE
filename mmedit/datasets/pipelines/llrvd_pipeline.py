import numpy as np
import torch
from ..registry import PIPELINES

def to_tensor(lq, gt):
    # [T, H, W, C] -> [T, C, H, W]
    # lq = torch.from_numpy(lq.transpose(0, 3, 1, 2).copy())
    # gt = torch.from_numpy(gt.transpose(0, 3, 1, 2).copy())
    
    lq = torch.from_numpy(np.ascontiguousarray(lq.transpose(0, 3, 1, 2)))
    gt = torch.from_numpy(np.ascontiguousarray(gt.transpose(0, 3, 1, 2)))

    return lq, gt


def rescale(lq, gt, ratio):
    black_level = 512
    white_level = 15360

    lq = lq.astype(np.float32)
    lq = (lq - black_level) / (white_level - black_level)
    lq = np.clip(lq * ratio, 0.0, 1.0)

    gt = gt.astype(np.float32)
    gt = np.float32((gt - black_level) / (white_level - black_level))
    gt = np.clip(gt, 0.0, 1.0)

    return lq, gt


@PIPELINES.register_module()
class LLRVDTrainPipeline:
    def __init__(self, crop_size, flip_ratio=[0.5, 0.5, 0.5], transpose_ratio=0.5, rescale_ratio=1.0):
        self.crop_size = crop_size
        self.flip_ratio = flip_ratio
        self.transpose_ratio = transpose_ratio
        self.rescale_ratio = rescale_ratio

    def crop(self, lq, gt, num_input_frames, sequence_length):
        data_h, data_w = lq.shape[1], lq.shape[2]
        crop_h, crop_w = self.crop_size

        start_frame = np.random.randint(0, sequence_length - num_input_frames + 1)
        h_offset = np.random.randint(0, data_h - crop_h + 1)
        w_offset = np.random.randint(0, data_w - crop_w + 1)

        cropped_lq = lq[start_frame:start_frame+num_input_frames, h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]
        cropped_gt = gt[start_frame:start_frame+num_input_frames, h_offset:(h_offset+crop_h), w_offset:(w_offset+crop_w)]

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
        lq, gt = self.flip(cropped_lq, cropped_gt)
        lq, gt = self.transposeHW(lq, gt)
        ratio = float(results['key'].split('/')[1])
        lq, gt = rescale(lq, gt, ratio)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=start_frame)


@PIPELINES.register_module()
class LLRVDValPipeline:
    def __init__(self, crop_size, val_num):
        self.crop_size = crop_size
        self.val_num = val_num

    def crop(self, lq, gt, num_input_frames):
        data_h, data_w = lq.shape[1], lq.shape[2]
        crop_h, crop_w = self.crop_size

        start_frame = 0
        h_offset = (data_h//2) - (crop_h//2)
        w_offset = (data_w//2) - (crop_w//2)

        cropped_lq = lq[start_frame:start_frame+num_input_frames, h_offset:h_offset+crop_h, w_offset:w_offset+crop_w]
        cropped_gt = gt[start_frame:start_frame+num_input_frames, h_offset:(h_offset+crop_h), w_offset:(w_offset+crop_w)]

        return cropped_lq, cropped_gt, start_frame
    
    def __call__(self, results):
        lq, gt, start_frame = self.crop(results['lq'], results['gt'], self.val_num)
        ratio = float(results['key'].split('/')[1])
        lq, gt = rescale(lq, gt, ratio)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=start_frame)


@PIPELINES.register_module()
class LLRVDTestPipeline:
    def __init__(self):
        pass
        
    def __call__(self, results):
        ratio = float(results['key'].split('/')[1])
        lq, gt = rescale(results['lq'], results['gt'], ratio)
        lq, gt = to_tensor(lq, gt)

        return dict(lq=lq, gt=gt,
                    lq_path=results['lq_path'],
                    gt_path=results['gt_path'],
                    key = results['key'],
                    start_frame=0)