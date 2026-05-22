# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
class WHU_MARS(BaseImageDataset):
    """
    WHU-MARS
    Reference:
    Zhao et al. WHU-MARS: A Multispectral Aerial-Ground Benchmark Towards Any-Scenario Person Re-Identification. CVPR 2026.
    URL: https://github.com/msm8976/WHU-MARS

    Dataset statistics:
    # identities: 1,000/2,337
    # images: 185,922/434,620
    """
    dataset_dir = 'WHU-MARS'
    # dataset_dir = 'WHU-MARS_2337'

    def __init__(self, root='', verbose=True, pid_begin = 0, modalities=None, **kwargs):
        super(WHU_MARS, self).__init__()
        if modalities is None:
            raise ValueError("DATASETS.MODALITIES are not be set.")
        self.modality_ls = list(modalities)
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> WHU-MARS loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    
    def _process_dir(self, dir_path, relabel=False):
        dataset = dict()
        pid_container = set()
        pattern = re.compile(r'(\d+)_c(\d+)')

        for modality_name in self.modality_ls:
            modality_dir = osp.join(dir_path, modality_name)
            img_paths = glob.glob(osp.join(modality_dir, '*.jpg'))
            img_paths = sorted(img_paths)
            for img_path in img_paths:
                basename = osp.basename(img_path)
                pid, _ = map(int, pattern.search(basename).groups())
                pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        
        for mindex, modality_name in enumerate(self.modality_ls, 1):
            modality_dir = osp.join(dir_path, modality_name)
            img_paths = glob.glob(osp.join(modality_dir, '*.jpg'))
            img_paths = sorted(img_paths)
            
            for img_path in img_paths:
                basename = osp.basename(img_path)
                pid, camid = map(int, pattern.search(basename).groups())
                # for WHU-MARS-1000-GD
                # if pid<1000 or pid>=2000 or camid>5:
                #     continue
                camid -= 1
                if relabel:
                    pid = pid2label[pid]
                dataset.setdefault(modality_name, []).append((img_path, pid, camid, mindex))
        return dataset
