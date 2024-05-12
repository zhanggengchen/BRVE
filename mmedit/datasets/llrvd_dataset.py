# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import numpy as np
import copy
from collections import defaultdict
import torch.distributed as dist

import mmcv
import cv2

from .base_dataset import BaseDataset
from .registry import DATASETS

black_level = 512
white_level = 15360

def Load_Pack(path, ratio=1.0, pack_order='RG1G2B'):
    img_paths = glob.glob(osp.join(path, "*.tiff"))
    img_paths.sort(key = lambda x: int(osp.splitext(osp.split(x)[-1])[0]))
    img_list = []
    
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32)
        img = (img - black_level) / (white_level - black_level)
        
        img = np.clip(img * ratio, 0.0, 1.0)
        
        # pack, R, G1, G2, B
        H, W = img.shape
        if pack_order == 'RG1G2B':
            img = np.stack((img[0:H:2, 0:W:2],
                            img[0:H:2, 1:W:2],
                            img[1:H:2, 0:W:2],
                            img[1:H:2, 1:W:2]), axis=2)
        else:
            img = np.stack((img[1:H:2, 0:W:2],
                            img[0:H:2, 0:W:2],
                            img[0:H:2, 1:W:2],
                            img[1:H:2, 1:W:2]), axis=2)
        
        img_list.append(img)
        
    return np.stack(img_list, axis=0)


@DATASETS.register_module()
class LLRVDDataset(BaseDataset):
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
                 ann_file,
                 memorize=True,
                 pack_order='RG1G2B',
                 num_input_frames=None,
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
        print("Raw data is packed in ", pack_order, "order")
        self.pack_order = pack_order

        self.lq_data = dict()
        self.gt_data = dict()
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for the dataset.

        Returns:
            list[dict]: Returned list of dicts for paired paths of LQ and GT.
        """

        data_infos = []
        ann_list = mmcv.list_from_file(self.ann_file)
        
        # 确保能够加载num_input_frames帧
        # for ann in ann_list:
        #     scene = ann.strip()
        #     gt_path = osp.join(self.gt_folder, scene)
        #     sequence_length = len(glob.glob(osp.join(gt_path, "*.tiff")))
        #     if self.num_input_frames is None:
        #         self.num_input_frames = sequence_length
        #     else:
        #         self.num_input_frames = min(self.num_input_frames, sequence_length)
        
        for ann in ann_list:
            scene = ann.strip()

            lq_scene_path = osp.join(self.lq_folder, scene)
            gt_path = osp.join(self.gt_folder, scene)
            
            # sequence_length = len(glob.glob(osp.join(gt_path, "*.tiff")))
            ratios = [r.split('/')[-1] for r in glob.glob(osp.join(lq_scene_path, "*"))]
            
            num_input_frames = self.num_input_frames
            
            for ratio in ratios:
                lq_path = osp.join(lq_scene_path, ratio)
                key = scene + "/" + ratio
                self.lq_data[lq_path] = None
                self.gt_data[gt_path] = None
                data_infos.append(
                    dict(lq=None,
                        gt=None,
                        lq_path=lq_path,
                        gt_path=gt_path,
                        key=key,
                        num_input_frames=num_input_frames))
        
        return data_infos
    
    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])

        lqp = self.data_infos[idx]['lq_path']
        gtp = self.data_infos[idx]['gt_path']

        if self.memorize:
            if self.lq_data[lqp] is None:
                print(f'{dist.get_rank()}/{dist.get_world_size()} loading {lqp}')
                self.lq_data[lqp] = np.load(osp.join(lqp, "lq.npy"), mmap_mode='r')
            
            if self.gt_data[gtp] is None:
                self.gt_data[gtp] = np.load(osp.join(gtp, "gt.npy"), mmap_mode='r')

            results['lq'] = self.lq_data[lqp]
            results['gt'] = self.gt_data[gtp]
        else:
            # print("....")
            results['lq'] = np.load(osp.join(lqp, "lq.npy"), mmap_mode='r')
            results['gt'] = np.load(osp.join(gtp, "gt.npy"), mmap_mode='r')

        results['sequence_length'] = results['gt'].shape[0]
            
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])

        lqp = self.data_infos[idx]['lq_path']
        gtp = self.data_infos[idx]['gt_path']

        if self.memorize:
            if self.lq_data[lqp] is None:
                print(f'{dist.get_rank()}/{dist.get_world_size()} loading {lqp}')
                self.lq_data[lqp] = np.load(osp.join(lqp, "lq.npy"), mmap_mode='r')
            
            if self.gt_data[gtp] is None:
                self.gt_data[gtp] = np.load(osp.join(gtp, "gt.npy"), mmap_mode='r')

            results['lq'] = self.lq_data[lqp]
            results['gt'] = self.gt_data[gtp]
        else:
            print("....")
            results['lq'] = np.load(osp.join(lqp, "lq.npy"), mmap_mode='r')
            results['gt'] = np.load(osp.join(gtp, "gt.npy"), mmap_mode='r')
        
        results['sequence_length'] = results['gt'].shape[0]
            
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
        
        ratio_result = {'100': defaultdict(list), "125": defaultdict(list), "160": defaultdict(list),
                        "200": defaultdict(list), "250": defaultdict(list), "320": defaultdict(list)}

        for res in results:
            ratio = res['gain']
            for metric, val in res['eval_result'].items():
                ratio_result[ratio][metric].append(val)

        for ratio in ratio_result.keys():
            ratio_result[ratio] = {
                metric: np.mean(val_list)
                for metric, val_list in ratio_result[ratio].items()
            }

        # average the results
        eval_result = defaultdict(list)
        for ratio in ratio_result.keys():
            for metric, val in ratio_result[ratio].items():
                eval_result[metric].append(val)

        eval_result = {
            metric: np.mean(val_list)
            for metric, val_list in eval_result.items()
        }

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