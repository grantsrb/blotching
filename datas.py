import torch
from envs import ProbGen

class DataIterable:
    def __init__(self, data, endx=None, batch_size=128):
        """
        Args:
            data: torch tensor (N, S)
                N is total samples, S is seq len
            endx: int or None
                if an int is argued, it should denote the last row
                of the data tensor that contains data. If None, it is
                assumed that the data tensor is full
            batch_size: int
                the batch size
        """
        self.data = data
        self.batch_size = batch_size
        self.idx = 0
        if endx is None: endx = len(data)
        self.perm = torch.randperm(endx).long()

    def __len__(self):
        return len(self.perm)//self.batch_size

    def __next__(self):
        """
        Returns:
            dict
                keys: str
                vals: tensor (bsize, seq_len-1)
                    input_ids: torch tensor (bsize, seq_len-1)
                    output_ids: torch tensor (bsize, seq_len-1)
        """
        if self.idx < self.__len__():
            strt = self.idx*self.batch_size
            end = strt + self.batch_size
            self.idx += 1
            sampls = self.data[self.perm[strt:end]]
            return {
                "input_ids": sampls[:,:-1],
                "output_ids": sampls[:,1:],
            }
        raise StopIteration

class DataCache(torch.utils.data.Dataset):
    """
    Handles storing the data
    """
    def __init__(self, max_samples=100000,
                       seq_len=100,
                       batch_size=128,
                       init_data=None,
                       dtype="long",
                       *args, **kwargs):
        """
        max_samples: int
            the maximum number of data samples for the data cache
        seq_len: int
            the maximum length of the data sequences
        batch_size: int
            the size of the batches for iteration
        init_data: torch Tensor (B, S) or None
            optional initial data
        """
        self.batch_size = batch_size
        self.cache = torch.zeros(max_samples, seq_len)
        if dtype=="long": self.cache = self.cache.long()
        self.is_full = False
        self.idx = 0
        if init_data is not None:
            self.add_data(init_data)

    @property
    def max_samples(self):
        return len(self.cache)

    @property
    def seq_len(self):
        return self.cache.shape[1]

    @property
    def n_loops(self):
        """
        Returns the number of iteration loops
        """
        return self.__len__()//self.batch_size

    def add_data(self, new_data):
        """
        new_data: torch tensor (B, S)
            a batch of new data. if the total amount of data exceeds
            self.max_samples, the oldest data will be replaced first.
        """
        sl = min(new_data.shape[1], self.seq_len)
        if len(new_data) < self.max_samples-self.idx:
            strt = self.idx
            self.idx = strt+len(new_data)
            self.cache[strt:self.idx, :sl] = new_data[:,:sl]
        elif len(new_data) > self.max_samples:
            self.cache[:,:sl] = new_data[:self.max_samples, :sl]
            self.idx = 0
            self.is_full = True
        else:
            self.is_full = True
            cram = self.max_samples-self.idx
            self.cache[self.idx:, :sl] = new_data[:cram, :sl]
            self.idx = len(new_data)-cram
            self.cache[:self.idx, :sl] = new_data[cram:, :sl]
        if self.idx >= self.max_samples:
            self.is_full = True
            self.idx = 0

    def __len__(self):
        if self.is_full: return self.max_samples
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
        self.iter_idx = 0
        endx = len(self.cache) if self.is_full else self.idx
        return DataIterable(
            self.cache.clone(), endx=endx, batch_size=self.batch_size
        )

class Tokenizer:
    def __init__(self, str2idx:dict, delimeters=[""], pad="P", null=" "):
        """
        Args:
            str2idx: dict
                keys: str
                    the string token values
                vals: int
                    the indices corresponding to the tokens
            delimeters: list of str
                a list of delimeters for the tokenization. It's much
                faster if you do a single delimeter of "".
            pad: str
                the padding token. currently will break if the padding
                token is longer than a single character.
            null: str
                the null token. currently will break if the null
                token is longer than a single character.
        """
        assert len(pad)==1 and len(null)==1
        self.pad = pad
        self.null = null
        self.delimeters = sorted(delimeters, key=lambda x: -len(x))
        self.str2idx = {**str2idx}
        self.idx2str = {v:k for k,v in str2idx.items()}
        self.pad_idx = self.str2idx[self.pad]
        self.null_idx = self.str2idx[self.null]

    @property
    def n_tokens(self):
        return len(self.str2idx)

    def tokenize(self, string, delims=[""], delim_idx=0):
        """
        Recursively splits the string into tokens separated by the
        delimeters.

        Args:
            string: str
                the raw string to be tokenized
            delims: list of str
                the delimeters. must be sorted from greatest length
                to shortest to ensure all delimeters are included
            delim_idx: int
                the current level of recursion
        Returns:
            splt: list of str
                a list of tokens not yet converted to their indices
        """
        if delims[delim_idx]=="":
            return list(string)
        splt = string.split(delims[delim_idx])
        delim_idx += 1
        if delim_idx < len(delims):
            chunks = None
            for s in splt:
                s = self.tokenize(s, delims, delim_idx)
                if chunks: chunks += s
                else: chunks = s
            splt = chunks
        return splt

    def idxs_to_strs(self, idxs):
        """
        Converts a list of indices to a list of stings

        Args:
            idxs: int or list of ints or tensor
                the indices to be converted to string values
        Returns:
            strings: list of str
                a list of the joined string values of the argued indices
        """
        if type(idxs)==int: idxs = [idxs]
        elif hasattr(idxs, "shape") and len(idxs.shape)==1: idxs = [idxs]
        strings = []
        for idx in idxs:
            try:
                if len(idx)>0:
                    strings.append(
                        "".join([ self.idx2str[int(i)] for i in idx ])
                    )
            except:
                print(idxs)
                print()
                print(idx)
                print()
                print()
        return strings
    
    def decode(self, idxs):
        """
        Converts a list of indices to a list of stings

        Args:
            idxs: int or list of ints or tensor
                the indices to be converted to string values
        Returns:
            strings: list of str
                a list of the joined string values of the argued indices
        """
        return self.idxs_to_strs(idxs)

    def str_to_idxs(self,
                    strings,
                    as_tensor=False,
                    max_len=None,
                    pad=False):
        """
        Converts a list of strings to a list of token index lists

        Args:
            strings: str or list of str
                the strings to be tokenized
            as_tensor: bool
                if true, will return indices as a pytorch long tensor
            max_len: int or None
                optional argument to truncate/pad the indexes
            pad: bool
                if true, will pad the index lists with pad indices
        Returns:
            idxs: list of ints
                a list of the integer indices of each token in the
                argued strings
        """
        if type(strings)==str: strings = [strings]
        idxs = []
        for s in strings:
          idxs.append([
            self.str2idx[s]for s in self.tokenize(s,self.delimeters,0)
          ])
        if (as_tensor or pad) and not max_len:
            max_len = max([len(i) for i in idxs])
        if max_len or pad:
            p = self.pad_idx
            padded = []
            for i,x in enumerate(idxs):
                if len(x)>max_len: px = x[:max_len]
                else: px = x+[p for _ in range(max_len-len(x))]
                idxs[i] = px
        if as_tensor:
            return torch.LongTensor(idxs)
        return idxs

    def __call__(self, strings, as_tensor=False, max_len=None, pad=False):
        """
        Converts a list of strings to a list of tokens

        Args:
            strings: str or list of str
                the strings to be tokenized
            as_tensor: bool
                if true, will return indices as a pytorch long tensor
            max_len: int or None
                optional argument to truncate/pad the indexes
            pad: bool
                if true, will pad the index lists with pad indices
        Returns:
            idxs: list of ints or LongTensor
                a list of the integer indices of each token in the
                argued strings
        """
        return self.str_to_idxs(
            strings, as_tensor=as_tensor, max_len=max_len, pad=pad
        )

def get_tokenizer(digit_embs=True, max_num=1000):
    """
    Creates and returns a tokenizer object.

    Args:
        digit_embs: bool
            if true, will only use digits 0-9 as embeddings. Otherwise
            will use a unique embedding for each possible numeric value.
        max_num: int
            the maximum possible numeric value. unnecessary if
            digit_embs is true.
    Returns:
        tokenizer: Tokenizer
    """
    delimeters = [""]
    if digit_embs: max_num = 10
    else: raise NotImplemented #Need to group digits together in tokenizer
    pad =  "P"
    null = " "
    str2idx={
        pad: 0, # Padding token
        null: 1, # Null token
        "=":  2,
        "+":  3,
        "*":  4,
        "-":  5,
        "/":  6,
        **{str(i):6+i for i in range(max_num)}
    }
    tokenizer = Tokenizer(
        str2idx=str2idx, pad=pad, null=null
    )
    return tokenizer

def sample_data(math_env,
                tokenizer,
                n_samples=100000,
                max_len=100):
    """
    Samples the raw string data from the math_env and then tokenizes.
    Returns LongTensor of tokens

    Args:
        math_env: ProbGen object
            this is the math problem generator object
        tokenizer: Tokenizer object
            if None, this function will create and return a tokenizer
            object
        n_samples: int
            the number of data points to sample
        max_len: int
            the max sequence length
    Returns:
        data: torch LongTensor (n_samples, max_len)
    """
    data = torch.zeros(n_samples, max_len)
    solns = []
    for i in range(n_samples):
        prob = ProbGen.sample_prob(
            max_num=math_env.max_num,
            max_ents=math_env.max_ents,
            p_mult=math_env.p_mult,
            space_mults=math_env.space_mults
        )
        soln = ProbGen.find_soln(prob)
        solns.append(soln)
        #indices = tokenizer(soln, as_tensor=True, max_len=max_len)
        #data[i] = indices[0]
    data = tokenizer(solns, as_tensor=True, max_len=max_len)
    return data


