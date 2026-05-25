import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class BucketBatchSampler(Sampler):
    
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True, bucket_size=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        
        self.buckets = []
        indices = np.arange(len(dataset))
        for i in range(0, len(indices), self.bucket_size):
            self.buckets.append(indices[i:i + self.bucket_size])
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.buckets)
            for bucket in self.buckets:
                np.random.shuffle(bucket)
        
        batches = []
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size