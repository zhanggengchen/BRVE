# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import numpy as np
import copy
from collections import defaultdict
import torch.distributed as dist

import mmcv

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module()
class LLSMOIDnpyDataset(BaseDataset):
    """General dataset for video super resolution, used for recurrent networks.

    The dataset loads several LQ (Low-Quality) frames and GT (Ground-Truth)
    frames. Then it applies specified transforms and finally returns a dict
    containing paired data and other information.

    This dataset takes an annotation file specifying the sequences used in
    training or test. If no annotation file is provided, it assumes all video
    sequences under the root directory is used for training or test.

    In the annotation file (.txt), each line contains:

        1. folder name;
        2. number of frames in this sequence (in the same folder)

    Examples:

    ::

        calendar 41
        city 34
        foliage 49
        walk 47

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (list[dict | callable]): A sequence of data transformations.
        ann_file (str): The path to the annotation file. If None, we assume
            that all sequences in the folder is used. Default: None
        num_input_frames (None | int): The number of frames per iteration.
            If None, the whole clip is extracted. If it is a positive integer,
            a sequence of 'num_input_frames' frames is extracted from the clip.
            Note that non-positive integers are not accepted. Default: None.
        test_mode (bool): Store `True` when building test dataset.
            Default: `True`.
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 gain,
                 ann_file,
                 num_input_frames=None,
                 memorize=True,
                 test_mode=True):
        super().__init__(pipeline, test_mode)

        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = ann_file
        self.memorize = memorize

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames
        self.sequence_length = 300

        assert gain in ['Gain0Gain0', 'Gain15Gain0', 'Gain30Gain0', 'all'], f'{gain} should in [Gain0Gain0, Gain15Gain0, Gain30Gain0, all]'
        self.gain = gain

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            list[dict]: Returned list of dicts for paired paths of LQ and GT.
        """

        data_infos = []
        ann_list = mmcv.list_from_file(self.ann_file)
        for ann in ann_list:
            lq_path, gt_path = ann.strip().split(' ')

            if (self.gain != 'all') and (not self.gain in lq_path):
                continue

            key = lq_path.split('/')[0] + '/' + lq_path.split('/')[1] # 'Scene/Gain'

            lq_path = osp.join(self.lq_folder, lq_path)
            gt_path = osp.join(self.gt_folder, gt_path)

            if self.num_input_frames is None:
                num_input_frames = self.sequence_length
            else:
                num_input_frames = self.num_input_frames

            data_infos.append(
                dict(lq=None,
                     gt=None,
                     lq_path=lq_path,
                     gt_path=gt_path,
                     key=key,
                     num_input_frames=num_input_frames,
                     sequence_length=self.sequence_length))
        
        return data_infos
    
    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        if self.memorize:
            if self.data_infos[idx]['lq'] is None:
                lqp = self.data_infos[idx]['lq_path']
                print(f'{dist.get_rank()}/{dist.get_world_size()} loading {lqp}')
                self.data_infos[idx]['lq'] = np.load(self.data_infos[idx]['lq_path']).transpose(0, 2, 3, 1)
            
            if self.data_infos[idx]['gt'] is None:
                self.data_infos[idx]['gt'] = np.load(self.data_infos[idx]['gt_path']).transpose(0, 2, 3, 1)

            return self.pipeline(self.data_infos[idx])
        else:
            results = copy.deepcopy(self.data_infos[idx])
            results['lq'] = np.load(self.data_infos[idx]['lq_path'], mmap_mode='r').transpose(0, 2, 3, 1)
            results['gt'] = np.load(self.data_infos[idx]['gt_path'], mmap_mode='r').transpose(0, 2, 3, 1)
            return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        if self.memorize:
            if self.data_infos[idx]['lq'] is None:
                lqp = self.data_infos[idx]['lq_path']
                print(f'{dist.get_rank()}/{dist.get_world_size()} loading {lqp}')
                # 减少验证数据的内存占用，只load前10帧
                self.data_infos[idx]['lq'] = np.load(self.data_infos[idx]['lq_path']).transpose(0, 2, 3, 1)[:self.num_input_frames, ...]
            
            if self.data_infos[idx]['gt'] is None:
                self.data_infos[idx]['gt'] = np.load(self.data_infos[idx]['gt_path']).transpose(0, 2, 3, 1)[:self.num_input_frames, ...]
            
            return self.pipeline(self.data_infos[idx])
        else:
            results = copy.deepcopy(self.data_infos[idx])
            results['lq'] = np.load(self.data_infos[idx]['lq_path'], mmap_mode='r').transpose(0, 2, 3, 1)[:self.num_input_frames, ...]
            results['gt'] = np.load(self.data_infos[idx]['gt_path'], mmap_mode='r').transpose(0, 2, 3, 1)[:self.num_input_frames, ...]
            return self.pipeline(results)

    
    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')
        
        if self.gain == 'all':
            gain_list = ['Gain0Gain0', 'Gain15Gain0', 'Gain30Gain0']
        else:
            gain_list = [self.gain]
        metric_list = results[0]['eval_result'].keys()
        eval_result = {}
        for gain in gain_list:
            for metric in metric_list:
                eval_result[f"{gain}_{metric}"] = 0.0

        for res in results:
            for metric, val in res['eval_result'].items():
                eval_result[f"{res['gain']}_{metric}"] += val
        eval_result = { metric: val / (len(self) / len(gain_list)) for metric, val in eval_result.items()}

        return eval_result
    
    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)

        return self.prepare_train_data(idx)
    
    def clear_cache(self):
        for idx in range(self.__len__()):
            self.data_infos[idx]['lq'] = None
            self.data_infos[idx]['gt'] = None