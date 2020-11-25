import math
import random
from collections import Counter

import torch
from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


class MultitaskBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batches = {}
        for idx in torch.randperm(len(self.data_source)).tolist():
            task = self.data_source[idx]["task"]
            if task not in batches:
                batches[task] = []
            batches[task].append(idx)
            for task, batch in batches.items():
                if len(batch) == self.batch_size:
                    yield batch
                    batches[task] = []
        for batch in batches.values():
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            nums_samples = {}
            for idx in range(len(self.data_source)):
                task = self.data_source[idx]["task"]
                if task not in nums_samples:
                    nums_samples[task] = 0
                nums_samples[task] += 1
            return sum(
                num_samples // self.batch_size
                for num_samples in nums_samples.values()
            )
        else:
            return (
                (len(self.data_source) + self.batch_size - 1)
                // self.batch_size
            )
