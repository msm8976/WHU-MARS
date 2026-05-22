from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class PKMSampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances for each modality,
    so the batch size is P*K*M.
    data_source: list of (img_path, pid, camid, modality).
    """
    
    def __init__(self, data_source, batch_size, num_instances, modalities):
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(lambda: defaultdict(list))
        self.modality_ls = list(modalities)

        if self.data_source.keys():
            for m in self.modality_ls:
                for index, (_, pid, _, _) in enumerate(self.data_source[m]):
                    self.index_dic[pid][m].append(index)
            self.pids = sorted(list(self.index_dic.keys()))

            self.length = 0
            self.pid_max_count = dict()
            for pid in self.pids:
                max_num = 0
                for m in self.modality_ls:
                    cnt = len(self.index_dic[pid][m])
                    cnt = max(cnt, self.num_instances)
                    max_num = max(max_num, cnt)
                max_num -= max_num % self.num_instances
                self.pid_max_count[pid] = max_num
                self.length += max_num

    def __iter__(self):
        batch_idxs_dict = defaultdict(lambda: defaultdict(list))

        for pid in self.pids:
            need = self.pid_max_count[pid]
            for m in self.modality_ls:
                idxs = copy.deepcopy(self.index_dic[pid][m])
                cur = len(idxs)
                if cur >= need:
                    idxs = np.random.choice(idxs, size=need, replace=False)
                else:
                    k, r = divmod(need, cur)
                    extended_idxs = list(idxs) * k
                    if r > 0:
                        extended_idxs += random.sample(list(idxs), r)
                    idxs = extended_idxs

                random.shuffle(idxs)
                groups = [idxs[i:i+self.num_instances] 
                          for i in range(0, len(idxs), self.num_instances)]
                batch_idxs_dict[pid][m] = groups

        final_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                temp_idxs=[]
                for m in self.modality_ls:
                    group = batch_idxs_dict[pid][m].pop(0)
                    temp_idxs.append(group)
                final_idxs.extend(zip(*temp_idxs))
                if len(batch_idxs_dict[pid][self.modality_ls[0]]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
