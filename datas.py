import collections
import datasets
from ml_utils.utils import try_key
import torch
import os

class DataIterable:
    def __init__(self, data, batch_size=128):
        """
        Args:
            data: torch tensor (N, S)
                N is total samples, S is seq len
            batch_size: int
                the batch size
        """
        self.data = data
        self.batch_size = batch_size
        self.idx = 0
        self.perm = torch.randperm(len(data)).long()

    def __next__(self):
        """
        Returns:
            input_indices: torch tensor (bsize, seq_len-1)
            output_indices: torch tensor (bsize, seq_len-1)
        """
        if self.idx < self.idx_order//self.batch_size:
            strt = self.idx*self.batch_size
            end = strt + self.batch_size
            self.idx += 1
            sampls = self.data[self.perm[strt:end]]
            return {
                "inputs": sampls[:,:-1],
                "outputs": sampls[:,1:],
            }
        raise StopIteration

def DataCache(torch.utils.data.Dataset):
    """
    Handles storing the data
    """
    def __init__(self, max_sampls=100000,
                       seq_len=100,
                       batch_size=128,
                       init_data=None):
        """
        max_sampls: int
            the maximum number of data samples for the data cache
        seq_len: int
            the maximum length of the data sequences
        batch_size: int
            the size of the batches for iteration
        init_data: torch Tensor (B, S) or None
            optional initial data
        """
        self.cache = torch.zeros(max_sampls, seq_len)
        self.is_full = False
        self.idx = 0
        if init_data:
            self.cache[:len(init_data), :init_data.shape[1]] = init_data
            self.idx = len(init_data)

    @property
    def max_sampls(self):
        return len(self.cache)

    @property
    def seq_len(self):
        return self.cache.shape[1]

    def add_data(self, new_data):
        """
        new_data: torch tensor (B, S)
            a batch of new data. if the total amount of data exceeds
            self.max_sampls, the oldest data will be replaced first.
        """
        sl = min(new_data.shape[1], self.seq_len)
        if len(new_data) < self.max_sampls-self.idx:
            strt = self.idx
            self.idx = strt+len(new_data)
            self.cache[strt:self.idx, :sl] = new_data[:,:sl]
        elif len(new_data) > self.max_sampls:
            self.cache[:,:sl] = new_data[:self.max_sampls, :sl]
            self.idx = 0
            self.is_full = True
        else:
            self.is_full = True
            cram = self.max_sampls-self.idx
            self.cache[self.idx:, :sl] = new_data[:cram, :sl]
            self.idx = len(new_data)-cram
            self.cache[:self.idx, :sl] = new_data[cram:, :sl]
        if self.idx >= self.max_sampls:
            self.is_full = True
            self.idx = 0

    def __len__(self):
        if self.is_full: return self.max_sampls
        return self.idx

    def __getitem__(self, idx):
        """
        Returns a single data sample

        Args:
            idx: int
        Returns:
            sample: torch tensor (1, seq_len)
        """
        return {
            "input_indices": self.cache[idx:idx+1,:-1],
            "output_indices": self.cache[idx:idx+1,1:],
        }

    def __iter__(self):
        self.idx_order = torch.randperm(self.__len__()).long()
        self.iter_idx = 0
        return DataIterable(self.cache.clone(), self.batch_size)

def sample_initial_data(math_env,
                        n_sampls=100000,
                        max_len=100,
                        tokenizer=None):
    """
    Samples the raw string data from the math_env and then tokenizes.
    Returns LongTensor of tokens

    Args:
        math_env: ProbGen object
            this is the math problem generator object
        n_sampls: int
            the number of data points to sample
        max_len: int
            the max sequence length
        tokenizer: Tokenizer object or None
            if None, this function will create and return a tokenizer
            object
    Returns:
        data: torch LongTensor (n_sampls, max_len)
        tokenizer: Tokenizer
    """
    data = torch.zeros(n_sampls,
    for i in range(n_sampls):
        prob = ProbGen.sample_prob(
            max_num=math_env.max_num,
            max_ents=math_env.max_ents,
            p_mult=math_env.p_mult,
            space_mults=math_env.space_mults
        )
        soln = ProbGen.find_soln(prob)





