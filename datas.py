import time
import torch
import torch.multiprocessing as mp
import envs
import copy
import numpy as np
from tqdm import tqdm
import utils
import math

class DataIterable:
    def __init__(self, data, batch_size=128, meta_data=None):
        """
        Args:
            data: torch tensor (N, S)
                N is total samples, S is seq len
            batch_size: int
                the batch size
            meta_data: dict of sequences or None
                optional additional dict of meta data for each row
                in data
        """
        self.data = data
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.idx = 0
        self.perm = torch.randperm(len(self.data)).long()

    def __len__(self):
        return int(np.ceil(len(self.perm)/self.batch_size))

    def __iter__(self):
        return self

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
            data = {
                "input_ids": sampls[:,:-1],
                "output_ids": sampls[:,1:],
            }
            if self.meta_data:
                data["meta_data"] = {}
                idxs = self.perm[strt:end].cpu().data.numpy()
                for k in self.meta_data:
                    data["meta_data"][k] = self.meta_data[k][idxs]
            return data
        raise StopIteration

class DataCache(torch.utils.data.Dataset):
    """
    Handles storing the data. Stores data as a single tensor. Then
    returns data as a dict with keys "input_ids" and "output_ids"
    """
    def __init__(self, max_samples=100000,
                       seq_len=100,
                       batch_size=128,
                       init_data=None,
                       dtype="long",
                       prob_len=None,
                       meta_data=None,
                       pad_id=0,
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
        prob_len: int or None
            the index location at which the problem ends and the
            solution begins. This is useful to speed up operations.
        meta_data: dict of lists (optional)
            meta data corresponding to each row in the data
        pad_id: int
            the default padding id
        """
        self.batch_size = batch_size
        if not max_samples and init_data is not None:
            max_samples = len(init_data)
        self.pad_id = pad_id
        self.cache = torch.full((max_samples, seq_len), self.pad_id)
        if dtype=="long": self.cache = self.cache.long()
        self.is_full = False
        self.idx = 0
        if init_data is not None:
            # Currently we need meta_data to be None to use add_data
            # function
            self.meta_data = None
            self.add_data(init_data)
        self.meta_data = meta_data
        self.prob_len = prob_len

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
        meta_data: dict of sequences or None
            the rows of meta_data should align with the rows of the
            cache
        """
        if self.meta_data: raise NotImplemented
        sl = min(new_data.shape[1], self.seq_len)
        if len(new_data) < self.max_samples-self.idx:
            strt = self.idx
            self.idx = strt+len(new_data)
            self.cache[strt:self.idx, :sl] = new_data[:,:sl]
            if sl<self.seq_len:
                self.cache[strt:self.idx,sl:] = self.pad_id
        elif len(new_data) > self.max_samples:
            self.cache[:,:sl] = new_data[:self.max_samples, :sl]
            if sl<self.seq_len:
                self.cache[:,sl:] = self.pad_id
            self.idx = 0
            self.is_full = True
        else: # Need to wrap around
            self.is_full = True
            cram = self.max_samples-self.idx
            self.cache[self.idx:, :sl] = new_data[:cram, :sl]
            if sl<self.seq_len:
                self.cache[self.idx:,sl:] = self.pad_id
            self.idx = len(new_data)-cram
            self.cache[:self.idx, :sl] = new_data[cram:, :sl]
            if sl<self.seq_len:
                self.cache[:self.idx,sl:] = self.pad_id
        if self.idx >= self.max_samples:
            self.is_full = True
            self.idx = 0

    def __len__(self):
        if self.is_full: return self.cache.shape[0]
        return self.idx

    @property
    def shape(self):
        return self.cache.shape

    def __getitem__(self, idx):
        """
        Returns a single data sample

        Args:
            idx: int
        Returns:
            sample: torch tensor (1, seq_len)
        """
        data = {
            "input_ids": self.cache[idx:idx+1,:-1],
            "output_ids": self.cache[idx:idx+1,1:],
        }
        if self.meta_data:
            data["meta_data"] = {}
            for k in self.meta_data:
                d = self.meta_data[k][idx:idx+1]
                data["meta_data"][k] = d
        return data

    def slice(self, idx, endx):
        """
        Returns a data slice

        Args:
            idx: int
                the starting index for the slice
            endx: int
                the ending index not inclusive
        Returns:
            sample: torch tensor (endx-idx, seq_len)
        """
        if idx>self.idx and not self.is_full: raise Exception
        elif endx>self.idx and not self.is_full:
            endx = self.idx
        data = {
            "input_ids": self.cache[idx:endx,:-1],
            "output_ids": self.cache[idx:endx,1:],
        }
        if self.meta_data:
            data["meta_data"] = {}
            for k in self.meta_data:
                d = self.meta_data[k][idx:endx]
                data["meta_data"][k] = d
        return data

    def __iter__(self):
        self.iter_idx = 0
        if self.is_full:
            iterable = DataIterable(
                self.cache.clone(),
                meta_data=self.meta_data,
                batch_size=self.batch_size
            )
        else:
            iterable = DataIterable(
                self.cache[:self.idx].clone(),
                meta_data=self.meta_data,
                batch_size=self.batch_size
            )
        return iterable

class Tokenizer:
    @staticmethod
    def get_tokenizer(digit_embs=True, max_num=1000, *args, **kwargs):
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
        sep = "="# Semantic separator
        eos = "E"
        tok_list = [
            pad, null, sep, eos, "+", "*", "/", "-",
        ]

        special_tokens = { t:i for i,t in enumerate(tok_list) }
        str2idx={
            **special_tokens,
            **{str(i):len(special_tokens)+i for i in range(max_num)}
        }
        tokenizer = Tokenizer(
          str2idx=str2idx, pad=pad, null=null, eos=eos, sep=sep
        )
        return tokenizer

    def __init__(self,
                 str2idx:dict,
                 delimeters=[""],
                 pad="P",
                 null=" ",
                 sep="|",
                 eos="E"):
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
            sep: str
                the separation token. This token separates the problem
                from the solution. currently will break if the sep
                token is longer than a single character.
            eos: str
                the eos token. currently will break if the eos
                token is longer than a single character.
        """
        assert len(pad)==1 and len(null)==1
        self.pad = pad
        self.null = null
        self.eos = eos
        self.sep = sep
        self.delimeters = sorted(delimeters, key=lambda x: -len(x))
        self.str2idx = {**str2idx}
        self.idx2str = {v:k for k,v in str2idx.items()}
        self.pad_idx = self.str2idx[self.pad]
        self.null_idx = self.str2idx[self.null]
        self.sep_idx = self.str2idx[self.sep]
        self.eos_idx = self.str2idx[self.eos]

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
            if len(idx)>0:
                strings.append(
                    "".join([ self.idx2str[int(i)] for i in idx ])
                )
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

    def strs_to_idxs(self,
                    strings,
                    as_tensor=False,
                    max_len=None,
                    pad=False,
                    add_eos=True):
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
            add_eos: bool
                if true, adds the eos token to the end of every
                string within strings
        Returns:
            idxs: list of ints
                a list of the integer indices of each token in the
                argued strings
        """
        if type(strings)==str: strings = [strings]
        idxs = []
        for stg in strings:
          iterator = self.tokenize(stg,self.delimeters,0)
          idxs.append( [self.str2idx[s] for s in iterator] )
        if add_eos:
            for i in range(len(idxs)):
                idxs[i].append(self.eos_idx)
        if (as_tensor or pad) and (not max_len or max_len<=0):
            max_len = max([len(i) for i in idxs])
        if max_len or pad:
            p = self.pad_idx
            padded = []
            for i,x in enumerate(idxs):
                if len(x)>=max_len: px = x[:max_len]
                else: px = x+[p for _ in range(max_len-len(x))]
                idxs[i] = px
        if as_tensor:
            return torch.LongTensor(idxs)
        return idxs

    def __call__(self,
                 strings,
                 as_tensor=False,
                 max_len=None,
                 pad=False,
                 add_eos=True):
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
            add_eos: bool
                if true, adds the eos token to the end of every
                string within strings
        Returns:
            idxs: list of ints or LongTensor
                a list of the integer indices of each token in the
                argued strings
        """
        # I'll allow it
        if type(strings)==type(torch.zeros(0)):
            return self.idxs_to_strs( strings )
        return self.strs_to_idxs(
            strings,
            as_tensor=as_tensor,
            max_len=max_len,
            pad=pad,
            add_eos=add_eos
        )

class Collector:
    """
    Handles the asynchronous data collection using the model.
    """
    def __init__(self, model, hyps, tokenizer):
        """
        Creates a deep copy of the model to be used asynchronously.

        Arguments:
            model: torch nn Module
                not the losswrapper, but the transformer or lstm model
            hyps: dict
                max_num: int
                    the maximum number available for sampling the initial
                    problem
                max_ents: int
                    maximum entities for the starting problem. If using
                    parentheticals, a parenthetical counts as one entity,
                    parentheticals are recursively samples with max_ents-1
                    max entities.
                p_mult: float [0,1]
                    the probability of sampling a multiplication sign for
                    the starting problem
                p_paren: float [0,1]
                    the probability of sampling a parenthetical.
                    Parentheticals are sampled the same way as the initial
                    problem but with max entities equal to max_ents-1
                space_mults: bool
                    if true, will not allow more than two numbers to be
                    multiplied together

                collection_size: int
                    the total number of new rollouts to collect
            tokenizer: Tokenizer
        Members:
            shared_exp: list of torch long tensors [(N,S)]
                a list of tensors that store data between processes.
                There is one shared tensor for each process
        """
        self.hyps = hyps
        self.n_procs = self.hyps["n_runner_procs"]
        self.model = copy.deepcopy(model)
        self.model.share_memory()
        self.tokenizer = tokenizer
        csize = self.hyps.get(
            "collection_size",
            hyps.get( "val_samples", int(0.1*hyps["max_samples"]) )
        )
        if hyps["exp_name"]=="test": csize = min(1000*self.n_procs, csize)
        self.shared_exp = [
            torch.zeros(
                int(csize//self.n_procs),
                self.hyps["seq_len"]
            ).long() for _ in range(self.n_procs)
        ]
        csize = self.shared_exp[0].shape[0]*self.n_procs
        self.hyps["collection_size"] = csize

        # Create gating mechanisms
        self.start_q = mp.Queue(self.n_procs)
        self.stop_q = mp.Queue(self.n_procs)
        self.terminate_q = mp.Queue(1)
        self.terminate_q.put(0)

        self.runners = []
        for r in range(self.n_procs):
            hyps = {**hyps, "seed": hyps["seed"]+r}
            runner = Runner(
                r,
                hyps=hyps,
                shared_exp=self.shared_exp[r],
                tokenizer=self.tokenizer,
                start_q=self.start_q,
                stop_q=self.stop_q,
                terminate_q=self.terminate_q,
            )
            self.runners.append(runner)

        self.procs = []
        self.init_runner_procs()

    def clear_experience(self):
        for exp in self.shared_exp:
            exp.zero_()

    def init_runner_procs(self):
        """
        Spawns the processes to actually collect the data
        """
        for i in range(self.n_procs):
            proc = mp.Process(
                target=self.runners[i].run,
                args=(self.model,)
            )
            proc.start()
            self.procs.append(proc)

    def await_runners(self):
        for i in range(self.n_procs):
            self.stop_q.get()

    def dispatch_runners(self):
        for i in range(self.n_procs):
            self.start_q.put(i)

    def terminate_procs(self):
        self.terminate_q.get()
        self.terminate_q.put(1)
        self.dispatch_runners()
        for proc in self.procs:
            proc.join()

    def harvest_exp(self):
        """
        Picks out the correct data from the shared tensors and returns
        a single tensor with all of the collected fresh data.

        Returns:
            exp: torch Tensor (N,S)
        """
        exp = []
        for i,tensor in enumerate(self.shared_exp):
            # -1 indicates the sample is wrong or long, so we ignore it
            t = tensor.clone()
            if self.hyps["exp_name"]=="test": # add some data if testing
                t[t==self.tokenizer.eos_idx] = self.tokenizer.pad_idx
                t[:5,-2] = self.tokenizer.eos_idx
                t[:5,-1] = self.tokenizer.pad_idx
                t[5:,-1] = -1
            exp.append(t[(t[:,-1]>=0)])
        exp = torch.cat(exp,dim=0)
        return exp

    def update_model(self, model):
        self.model.load_state_dict(model.state_dict())

class Runner:
    def __init__(self,
            idx,
            hyps,
            shared_exp,
            tokenizer,
            start_q,
            stop_q,
            terminate_q
        ):
        """
        This class handles the actual collection of the data in a
        separate process. When it fills the shared data tensor, it
        marks incorrect data with a -1 at the last index. Assume the
        data is correct in the absence of the -1

        Args:
            idx: int
                an integer id specific to this runner object
            hyps: dict
                the hyperparameters for the experiment
            shared_exp: shared torch Tensor (N,S)
            start_q: multiprocessing Queue.
                Allows main process to control when rollouts should be
                collected.
            stop_q: multiprocessing Queue.
                Used to indicate to main process that a rollout has
                been collected.
            phase_q: multiprocessing Queue.
                Used to indicate from the main process that the phase
                has changed.
            terminate_q: multiprocessing Queue.
                Used to indicate the end of the training from the main
                process.
        """
        self.idx = idx
        self.hyps = hyps
        self.prob_len = self.hyps["prob_len"]
        self.shared_exp = shared_exp
        self.tokenizer = tokenizer
        self.start_q = start_q
        self.stop_q = stop_q
        self.terminate_q = terminate_q

    def set_random_seed(self, seed):
        self.rand = np.random.default_rng(seed)

    def run(self, model):
        """
        run is the entry function to begin collecting rollouts from the
        environment. start_q indicates when to begin collecting a
        rollout and is controlled from the main process. The stop_q is
        used to indicate to the main process that a new rollout has
        been collected.
        """
        self.set_random_seed(self.hyps["seed"])
        self.model = model
        self.env = envs.MathEnv(**self.hyps)
        self.n_procs = self.hyps["n_runner_procs"]
        n_samps = self.shared_exp.shape[0]
        while True:
            with torch.no_grad():
                # Await collection signal from main proc
                rank = self.start_q.get()
                terminate = self.terminate_q.get()
                self.terminate_q.put(terminate)
                if terminate==1:
                    del self.shared_exp
                    del self.model
                    print("Terminating runner")
                    del self.start_q
                    del self.stop_q
                    del self.terminate_q
                    return None
                # Collect rollouts
                self.rollout(model=self.model, n_samps=n_samps)
                # Signals to main process that data has been collected
                self.stop_q.put(rank)

    def rollout(self, model, n_samps):
        """
        Uses the model to collect new data. Marks incorrect data with
        a -1 at the last index
        """
        probs = []
        solns = []
        soln_vals = []
        model.eval()
        self.shared_exp.zero_()
        device = model.get_device()
        for i in range(n_samps):
            prob = self.env.sample()
            probs.append(prob)

            val = envs.eval_prob(prob)
            soln_vals.append(str(val))

            soln = envs.MathEnv.find_soln(prob)
            solns.append(self.tokenizer.sep+soln)
        inpts = self.tokenizer(
            probs,
            as_tensor=True,
            max_len=self.prob_len+1, # Add a space for the first sep token
            pad=True,
            add_eos=False
        )
        inpts[:,-1] = self.tokenizer.sep_idx

        pad_mask = inpts==self.tokenizer.pad_idx
        bsize = self.hyps.get("val_batch_size", 100)
        plen = self.prob_len
        blotch_p = self.hyps.get("bootstrap_blotch_p", 0)
        if blotch_p == "variable": blotch_p = None
        with torch.no_grad():
            for i in range(0,len(inpts),bsize):
                startx = i
                endx = i+bsize
                ret_dict = model(
                    inpts[startx:endx].to(device),
                    pad_mask=pad_mask[startx:endx].to(device),
                    is_causal=True,
                    tforce=False,
                    n_steps=self.shared_exp.shape[1]-inpts.shape[1]-1,
                    temperature=self.hyps.get("val_temp", None),
                    incl_all_inpts=True,
                    blotch_p=blotch_p
                )
                logits = ret_dict["preds"]
                preds = torch.argmax(logits, dim=-1)
                preds[:,-1] = self.tokenizer.eos_idx
                ends = torch.argmax(
                  (preds[:,plen:]==self.tokenizer.eos_idx).long(),
                  dim=-1
                )
                strings = self.tokenizer.decode(preds[:,plen:])
                # Mark samples as incorrect if they're long or wrong
                for i,(pred,targ) in enumerate(zip(strings, soln_vals)):
                    # Check if solution is longer than ground truth
                    if ends[i] >= len(solns[i]):
                        preds[i,-1] = -1
                    else:
                        ans = pred.split(self.tokenizer.eos)[0]
                        ans = ans.split(self.tokenizer.sep)[-1]
                        # Check if final solution is incorrect
                        if ans != targ: preds[i,-1] = -1
                        elif plen+ends[i]+1<preds.shape[1]:
                            idx = plen+ends[i]+1
                            preds[i,idx:] = self.tokenizer.pad_idx
                self.shared_exp[startx:endx,:] = preds

def bootstrap_data(
        hyps,
        model,
        tokenizer,
        env=None,
        held_out_probs=set(),
        force_short=True,
        verbose=True
    ):
    """
    Uses the model to generate new data from randomly sampled problems.

    Args:
        hyps: dict
            val_batch_size: int
                the batch size for the augmentation loop
            aug_loops: int
                the number of augmentation loops to perform
        model: torch Module
        tokenizer: Tokenizer
        env: MathEnv or None
            if None, one is constructed using the argued hyps
        held_out_probs: set of str
            any problems that should be held out of the sampling
        force_short: bool
            if true, will only collect samples that have a shorter
            solution length than the ground truth.
    Returns:
        new_data: list of torch tensor [(A,S), (A1,S), ...]
            a list of variable length tensors that are the augmented
            samples.
    """
    bsize = hyps.get("val_batch_size", hyps["batch_size"])
    n_samps = bsize*hyps.get("star_loops", 1)
    if env is None: env = envs.MathEnv(**hyps)
    plen = hyps["prob_len"]
    probs = []
    new_samps = []
    eos = tokenizer.eos_idx
    pad = tokenizer.pad_idx
    sep = tokenizer.sep_idx
    model.eval()
    device = model.get_device()

    # First generate new problems and solutions
    # Add 2 for sep and eos tokens
    max_slen = len(str(env.get_max_val()))+2
    fin_solns = torch.zeros(n_samps, max_slen)+pad
    fin_solns[:,0] = sep
    soln_lens = torch.zeros(n_samps)
    solns = []
    for i in range(n_samps):
        prob = env.sample(held_out_probs=held_out_probs)
        probs.append(prob)

        ans = torch.LongTensor([
            tokenizer.str2idx[x] for x in str(envs.eval_prob(prob))
        ])
        fin_solns[i,1:1+len(ans)] = ans
        fin_solns[i,1+len(ans)] = eos

        soln = envs.MathEnv.find_soln(prob)
        solns.append(soln)
        soln_lens[i] = len(tokenizer.sep+soln)
    inpts = tokenizer(
        probs,
        as_tensor=True,
        max_len=plen+1, # Add a space for the first sep token
        pad=True,
        add_eos=False
    )
    inpts[:,-1] = tokenizer.sep_idx

    pad_mask = inpts==tokenizer.pad_idx
    aranges = torch.arange(hyps["seq_len"]).long()
    aranges = aranges[None].repeat((bsize,1)).to(device)
    blotch_p = hyps.get("bootstrap_blotch_p", 0)
    if blotch_p == "variable": blotch_p = None
    with torch.no_grad():
        for i in range(0,len(inpts),bsize):
            startx = i
            endx = i+bsize
            ret_dict = model(
                inpts[startx:endx].to(device),
                pad_mask=pad_mask[startx:endx].to(device),
                is_causal=True,
                tforce=False,
                n_steps=hyps["seq_len"]-inpts.shape[1]-1,
                temperature=hyps.get("aug_temp", None),
                incl_all_inpts=True,
                blotch_p=blotch_p
            )
            logits = ret_dict["preds"]
            preds = torch.argmax(logits, dim=-1)
            preds[:,-1] = tokenizer.eos_idx

            ends = torch.argmax(
              (preds[:,plen:]==tokenizer.eos_idx).long(),
              dim=-1
            )
            is_short = ends<soln_lens[startx:endx].to(device)
            if not force_short:
                is_short = torch.ones_like(is_short).bool()
            is_correct = utils.vectorized_check_correct(
                tokenizer, fin_solns[startx:endx].to(device), preds
            )
            preds[aranges>(plen+ends[:,None])] = pad
            samps =   preds[is_short&is_correct]
            new_samps.append(samps)
    return new_samps


def axe_data(
        hyps,
        wrapped_model,
        data_cache,
        tokenizer,
        in_place=False,
        verbose=True
    ):
    """
    Takes the data from a data cache and attempts to remove semantic
    segments by ablating steps and checking the resulting perplexity
    on the the continuation of the sample.

    Args:
        hyps: dict
            val_batch_size: int
                the batch size for the augmentation loop
            ablate_rel_tolerance: float
                the maximum change in loss to remove a step for a sample
        wrapped_model: LossWrapper
        data_cache: DataCache
        tokenizer: Tokenizer
        in_place: bool
            if true, will change samples in place, replacing them
            within the data cache. If False, will not replace in
            place but will still return a copy of the augmented samples.
    Returns:
        aug_data: list of torch tensor [(A,S), (A1,S), ...]
            a list of variable length tensors that are the augmented
            samples.
    """
    wrapped_model.eval()
    abs_tol = hyps.get("abs_axe_tol", math.inf)
    if not abs_tol: abs_tol = math.inf
    rel_tol = hyps.get("rel_axe_tol", math.inf)
    if not rel_tol: rel_tol = math.inf
    bsize = hyps.get("val_batch_size", 500)
    n_loops = hyps.get("axe_loops", 3)
    if n_loops < 0: n_loops = np.inf
    plen = data_cache.prob_len
    use_perplexity = hyps.get("axe_use_ppl", False)
    blotch_p = hyps.get("axe_blotch_p", None)
    if blotch_p is None: blotch_p = hyps.get("bootstrap_blotch_p", 0)
    device = wrapped_model.model.get_device()
    pad = tokenizer.pad_idx
    eos = tokenizer.eos_idx
    sep = tokenizer.sep_idx
    ans_len = len(str(hyps["max_val"]))

    aug_data = []
    # Aranges will help with indexing for padding and other things
    og_aranges = torch.arange(data_cache.shape[1])[None].repeat(
        (bsize, 1)
    ).to(device)
    og_barange = torch.arange(bsize)
    with torch.no_grad():
        perm = torch.randperm(len(data_cache)).long()
        m = min(n_loops, len(data_cache)//bsize)
        if m==0: m = 1
        rng = range(m)
        if verbose: rng = tqdm(rng)
        for i in rng:
            startx = i*bsize
            endx = startx+bsize
            permxs = perm[startx:endx] # Used later in the in_place op
            ids = data_cache.cache[permxs]
            aranges = og_aranges[:len(ids)]
            barange = og_barange[:len(ids)]
            outputs = ids[:,1:].to(device)
            inpts = ids[:,:-1].to(device)
            inpt_pad_mask = ((inpts==pad)|(inpts==eos)).to(device)
            out_pad_mask = (outputs==pad)
            data = {
                "input_ids": inpts,
                "output_ids": outputs,
                "inpt_pad_mask": inpt_pad_mask,
                "out_pad_mask": out_pad_mask.clone()
            }

            # First check if the model will solve the problems without
            # help
            ret_dict = wrapped_model(
                data,
                tforce=False,
                no_grad=True,
                prob_len=plen,
                ret_preds=True,
                temperature=hyps.get("aug_temp", None),
                blotch_p=blotch_p,
            )
            pred_ids = ret_dict["preds"]
            was_correct = utils.vectorized_check_correct(
                tokenizer, data["output_ids"], pred_ids
            )

            # Subtract 1 to remove because there will always be at
            # least a single separator
            sep_sums = torch.sum(inpts==sep, dim=-1)-1
            max_seps = torch.max(sep_sums)
            if max_seps==0: continue
            losses = []
            blosses = []
            masks = []
            for j in range(-1, max_seps):
                if j>-1:
                    blotch_mask = utils.get_blotch_mask(
                        inpts,
                        sep_idx=sep,
                        step_idx=j,
                    )
                    did_blotch = sep_sums>j
                    # TODO: Don't calculate unnecessary rows
                    data["inpt_pad_mask"] = inpt_pad_mask|blotch_mask
                    data["out_pad_mask"][:,:-1] = out_pad_mask[:,:-1]|\
                                                  blotch_mask[:,1:]
                ret_dict = wrapped_model(
                    data,
                    tforce=True,
                    no_grad=True,
                    prob_len=plen,
                    reduce_metrics=False,
                    ret_preds=True,
                    blotch_p=blotch_p,
                )
                loss = ret_dict["loss"].to(device) # Per Word Loss
                if use_perplexity:
                    loss = torch.exp(loss)

                if j==-1:
                    base_loss = loss
                else:
                    # only want to look at loss following the axing
                    omask = ~data["out_pad_mask"]
                    # Find index of current axing
                    ax = torch.argmax(blotch_mask.long(),dim=-1)
                    ax = (aranges<ax[:,None])[:,1:]
                    # Set all losses before ax to 0
                    loss[ax] = 0
                    omask[ax] = 0
                    base_loss[ax] = 0

                    div = omask.float().sum(-1)
                    bloss = base_loss.sum(-1)/div

                    loss = loss.sum(-1)/div
                    pred_ids = ret_dict["preds"]
                    pred_ids[:5] = data["output_ids"][:5]
                    is_correct = utils.vectorized_check_correct(
                        tokenizer, data["output_ids"], pred_ids
                    )
                    loss[~(was_correct&did_blotch&is_correct)] = math.inf
                    losses.append(loss)
                    blosses.append(bloss)

            losses = torch.stack(losses, dim=-1)
            blosses = torch.stack(blosses,dim=-1)
            abs_diffs = losses-blosses
            best_diffs, best_steps = torch.min(abs_diffs,dim=-1)
            div = blosses[barange,best_steps]
            best_rels = best_diffs/div
            idxs = ((best_diffs<abs_tol)&(best_rels<rel_tol)).cpu()
            if hyps["exp_name"]=="test": idxs[:10] = True
            if torch.any(idxs):
                new_data = ids[idxs]
                axe_idxs = utils.get_blotch_mask(
                    new_data,
                    sep_idx=sep,
                    step_idx=best_steps[idxs].cpu(),
                )
                new_data[axe_idxs] = tokenizer.pad_idx
                # Add data to list, and cache if in_place
                if len(new_data) > 0:
                    aug_data.append(new_data.cpu())
                    if in_place:
                        data_cache.cache[permxs[idxs], :] = new_data.cpu()
    return aug_data

def axe_step(ids, steps):
    """
    Removes the specified semantic step from the ids by replacing
    it with padding.

    Args:
        ids: torch LongTensor (B,N)
        steps: torch LongTensor (B,)
    Returns:
        rem_ids: torch LongTensor (B,N)
            the remaining ids after the axing
    """

def augment_data(
        hyps,
        model,
        data_cache,
        tokenizer,
        in_place=False,
        verbose=True
    ):
    """
    Takes the data from the data cache and creates augmentations by
    using the model to predict the sequence. Only stores/changes
    samples that are correct and shorter than before.

    Args:
        hyps: dict
            val_batch_size: int
                the batch size for the augmentation loop
            aug_loops: int
                the number of augmentation loops to perform
        model: torch Module
        data_cache: DataCache
        tokenizer: Tokenizer
        in_place: bool
            if true, will change samples in place, replacing them
            within the data cache. If False, will not replace in
            place but will still return a copy of the augmented samples.
    Returns:
        aug_data: list of torch tensor [(A,S), (A1,S), ...]
            a list of variable length tensors that are the augmented
            samples.
    """
    bsize = hyps.get("val_batch_size", 500)
    aug_loops = hyps.get("aug_loops", 3)
    if aug_loops < 0: aug_loops = np.inf
    plen = data_cache.prob_len
    device = model.get_device()
    pad = tokenizer.pad_idx
    eos = tokenizer.eos_idx
    sep = tokenizer.sep_idx
    ans_len = len(str(hyps["max_val"]))
    aug_data = []
    # Aranges will help with indexing for padding and other things
    aranges = torch.arange(data_cache.shape[1])[None].repeat(
        (bsize, 1)
    ).to(device)
    blotch_p = hyps.get("bootstrap_blotch_p", 0)
    if blotch_p == "variable": blotch_p = None
    with torch.no_grad():
        perm = torch.randperm(len(data_cache)).long()
        rng = range(min(aug_loops, len(data_cache)//bsize))
        if verbose: rng = tqdm(rng)
        for i in rng:
            startx = i*bsize
            endx = startx+bsize
            permxs = perm[startx:endx] # Used later in the in_place op
            data = data_cache.cache[permxs]
            inpts = data[:,:-1]
            pad_mask = inpts==pad
            outputs = data[:,1:].to(model.get_device())

            ret_dict = model(
                inpts[:,:plen+1].to(device),
                pad_mask=pad_mask[:,:plen+1].to(device),
                is_causal=True,
                tforce=False,
                # -2 for the plen+1 in inpts and inclusion of all inpts
                n_steps=data_cache.shape[1]-plen-2,
                incl_all_inpts=True,
                temperature=hyps.get("aug_temp", None),
                blotch_p=blotch_p
            )
            logits = ret_dict["preds"]
            preds = torch.argmax(logits, dim=-1)
            if hyps["exp_name"] == "test": 
                preds[:,1:] = outputs.clone()

            # Index of last relevant token. Guaranteed to be last index
            # or shorter because of preds[:,-1] = eos
            pred_ends = torch.argmax( (preds==eos).long(), dim=-1 )
            preds[pred_ends==0,-1] = eos
            pred_ends[pred_ends==0] = preds.shape[1]
            # Index of last relevant token
            soln_ends = torch.argmax( (outputs==eos).long(), dim=-1 )
            # Set anything that doesn't have an eos token to the last
            # index
            soln_ends[soln_ends==0] = outputs.shape[-1]-1
            shorts = ((soln_ends-pred_ends)>0)&(pred_ends>0)

            if hyps["exp_name"] == "test": 
                shorts[:5] = True

            if len(aranges)!=len(preds):
                aranges = aranges[:len(preds)]
            pad_idxs = aranges>pred_ends[:,None]
            preds[pad_idxs] = pad

            corrects = utils.vectorized_check_correct(
                tokenizer, outputs, pred_ids=preds[:,1:]
            )
            ## Find last separator and eos to determine window of answer.
            ## Then use sum of matching tokens to determine if correct.
            #seps = outputs==sep

            #last_sep_idxs = torch.argsort(
            #    seps.long(), dim=-1, descending=True
            #)[torch.arange(len(seps)).long(), seps.sum(-1).long()-1]
            #soln_ends = soln_ends[:,None]
            #soln_lens = soln_ends - last_sep_idxs[:,None]
            #soln_idxs =(aranges<soln_ends)&(aranges>=(soln_ends-soln_lens))
            #ans_idxs = (aranges < pred_ends[:,None])&\
            #           (aranges>=(pred_ends[:,None]-soln_lens))
            #corrects = torch.zeros_like(outputs).float()
            #idx = preds[ans_idxs]==outputs[soln_idxs[:,:-1]]
            #corrects[soln_idxs[:,:-1]] = (idx).float()
            #corrects = corrects.sum(-1)
            #corrects = corrects.squeeze()==soln_lens.squeeze()
            idx = shorts&corrects
            p = preds[idx].data.cpu()

            # Add data to list, and cache if in_place
            if len(p) > 0:
                aug_data.append(p)
                if in_place:
                    idx = idx.cpu()
                    data_cache.cache[permxs[idx], :] = p
    return aug_data


def sample_data(math_env,
                tokenizer,
                n_samples=100000,
                max_len=100,
                ret_strings=False,
                held_out_probs=set()):
    """
    Samples the raw string data from the math_env and then tokenizes.
    Returns LongTensor of tokens

    Args:
        math_env: MathEnv object
            this is the math problem generator object
        tokenizer: Tokenizer object
            if None, this function will create and return a tokenizer
            object
        n_samples: int
            the number of data points to sample
        max_len: int
            the max sequence length
        ret_strings: bool
            if true, will return string versions of problems and
            solutions
        held_out_probs: set of str
            any problems that should be held out of the sampling
    Returns:
        probs: torch LongTensor (n_samples, (max_digits+1)*max_ents-1)
        solns: torch LongTensor (n_samples, max_len-probs.shape[1])
        if ret_strings:
            prob_strs: list of str
                the problem strings
            soln_strs: list of str
                the solution strings
    """
    plen = math_env.prob_len
    if max_len is None:
        slen = math_env.max_soln_len+2 #+1 for eos, +1 for sep
    else: slen = max_len-plen
    assert slen>0, "Needs larger max_len!"

    prob_set = set()
    probs = []
    solns = []
    max_soln_len = math_env.max_soln_len+2
    n_overlapping = 0
    for i in range(n_samples):
        prob = math_env.sample()
        loop_count = 0
        while (prob in held_out_probs or prob in prob_set) and\
                                                loop_count<100:
            prob = math_env.sample()
            loop_count += 1
            if loop_count>=100: n_overlapping += 1
        prob_set.add(prob)
        probs.append(prob)
        soln = envs.MathEnv.find_soln(prob)
        solns.append(tokenizer.sep+soln)
        if len(solns[-1])+1>max_soln_len:
            max_soln_len = len(solns[-1])+1
    print("Overlapping Samples:", n_overlapping)
    if max_len is None and max_soln_len>slen: slen = max_soln_len

    prob_ids = tokenizer(probs,
        as_tensor=True,
        max_len=plen,
        pad=True,
        add_eos=False
    )
    soln_ids = tokenizer(solns,
        as_tensor=True,
        max_len=slen,
        pad=True,
        add_eos=True
    )
    if ret_strings:
        return prob_ids, soln_ids, probs, solns
    return prob_ids, soln_ids

def get_data_cache(math_env,
                   tokenizer,
                   init_samples=100000,
                   seq_len=100,
                   max_samples=None,
                   batch_size=128,
                   ret_strings=False,
                   held_out_probs=set(),
                   *args, **kwargs):
    """
    Creates an initial data_cache for storing and providing data.

    Args:
        math_env: MathEnv object
            this is the math problem generator object
        tokenizer: Tokenizer object
            if None, this function will create and return a tokenizer
            object
        init_samples: int
            the initial number of data points to sample
        seq_len: int
            the desired sequence lengths of the samples
        max_samples: int or None
            a parameter to limit the total amount of data in the cache.
            if None, will default to init_samples
        batch_size: int
            the batch_size for the DataCache iterable
        ret_strings: bool
            if true, will return string versions of problems and
            solutions
        held_out_probs: set of str
            any problems that should be held out of the sampling
    Returns:
        data_cache: DataCache
        if ret_strings:
            prob_strs: list of str
                the problem strings
            soln_strs: list of str
                the solution strings
    """
    if max_samples is None: max_samples = int(init_samples)
    prob_ids, soln_ids, probs, solns = sample_data(
        math_env,
        tokenizer,
        n_samples=int(init_samples),
        max_len=seq_len,
        ret_strings=True,
        held_out_probs=held_out_probs
    )
    prob_len = prob_ids.shape[1]
    data = torch.cat([prob_ids, soln_ids], dim=1)
    if seq_len is None: seq_len = data.shape[-1]
    data_cache = DataCache(
        max_samples=max_samples,
        seq_len=seq_len,
        batch_size=batch_size,
        init_data=data,
        prob_len=prob_len,
        pad_id=tokenizer.pad_idx,
    )
    if ret_strings:
        return data_cache, probs, solns
    return data_cache

def get_validation_set(
        math_env,
        tokenizer,
        max_len=100,
        batch_size=128,
        rand_samps=None,
        held_out_probs=set(),
        *args, **kwargs
    ):
    """
    Creates an validation data cache that includes problems of many
    varieties.

    Args:
        math_env: MathEnv object
            this is the math problem generator object
        tokenizer: Tokenizer object
            if None, this function will create and return a tokenizer
            object
        init_samples: int
            the initial number of data points to sample
        seq_len: int
            the desired sequence lengths of the samples
        max_samples: int or None
            a parameter to limit the total amount of data in the cache.
            if None, will default to init_samples
        batch_size: int
            the batch_size for the DataCache iterable
        rand_samps: int or None
            integer argument if you want to randomly sample n problems
            rather than systematically looking at all possible problems.
        held_out_probs: set of str
            any problems that should be held out of the sampling
    Returns:
        data_cache: DataCache
    """
    probs = []
    solns = []
    labels = []
    max_soln_len = 0
    if rand_samps:
        prob_set = set()
        for i in range(rand_samps):
            prob = math_env.sample()
            loop_count = 0
            while (prob in held_out_probs or prob in prob_set) and\
                                                    loop_count<100:
                prob = math_env.sample()
                loop_count += 1
            probs.append(prob)
            prob_set.add(prob)
    else:
        probs = envs.MathEnv.recursive_probs(
            prob="",
            n_ents=math_env.max_ents,
            max_num=math_env.max_num
        )
    print("Collecting Problem Solutions")
    for prob in tqdm(probs):
        soln, labs = envs.MathEnv.find_soln( prob, ret_labels=True )
        labels.append(tokenizer.sep.join(labs))
        solns.append( tokenizer.sep + soln )
        if len(solns[-1])>max_soln_len:
            max_soln_len = len(solns[-1])

    prob_ids = tokenizer(probs,
        as_tensor=True,
        max_len=math_env.prob_len,
        pad=True,
        add_eos=False
    )
    if max_len is None: slen = max_soln_len
    else: slen = max_len - math_env.prob_len
    soln_ids = tokenizer(
        solns,
        as_tensor=True,
        max_len=slen,
        pad=True,
        add_eos=True
    )
    prob_len = prob_ids.shape[1]
    data = torch.cat([prob_ids, soln_ids], dim=1)
    seq_len = data.shape[-1]
    data_cache = DataCache(
        max_samples=len(data),
        seq_len=seq_len,
        batch_size=batch_size,
        init_data=data,
        prob_len=prob_len,
        meta_data={
            "labels": np.asarray(labels, dtype="object"),
            "probs":  np.asarray(probs, dtype="object"),
            "solns":  np.asarray(solns, dtype="object")
        },
        pad_id=tokenizer.pad_idx,
    )
    return data_cache

def get_all_problems(math_env, shuffle=True):
    """
    Returns all possible problems for the given math environment.

    Args:
        math_env: MathEnv object
        shuffle: bool
            if true, will return a shuffled list of all possible
            problems, otherwise it is ordered
    Returns:
        all_probs: list of str
    """
    all_probs = envs.MathEnv.recursive_probs(
        prob="",
        n_ents=math_env.max_ents,
        max_num=math_env.max_num,
        mult=math_env.p_mult>0,
        space_mults=math_env.space_mults
    )
    if shuffle: np.random.shuffle(all_probs)
    return all_probs

def get_n_probs(math_env, n_probs, held_out_probs=set()):
    """
    Returns all possible problems for the given math environment.

    Args:
        math_env: MathEnv object
        n_probs: int
            the number of problems to sample
        held_out_probs: set of str
            a set of problems to not sample
    Returns:
        probs: list of str
    """
    probs = []
    prob_set = set()
    for i in range(n_probs):
        prob = math_env.sample()
        loop_count = 0
        while (prob in held_out_probs or prob in prob_set) and\
                                                loop_count<100:
            prob = math_env.sample()
            loop_count += 1
        probs.append(prob)
        prob_set.add(prob)
    return probs

def get_n_ood_probs(
        math_env,
        n_samps,
        min_num=None,
        min_mult_num=None,
        min_ents=None,
    ):
    """
    Returns a list of problems that will contain a non-mult
    number greater than min_num or a mult with a number greater than
    min_mult_num or both.

    Args:
        math_env: MathEnv
            a math env that is already set with the maximum values.
        n_samps: int
            the number of samples to return. will return fewer if
            impossible or low probability
        min_num: int or None
            a number denoting the minimum non-multiplied value that
            must be included in the problem unless a min_mult_num
            is included instead. if None, will be ignored, and will
            focus only on min_mult_num
        min_mult_num: int or None
            a number denoting the minimum multiplied value that
            must be included in the problem unless a min_num
            is included instead. if None, will be ignored, and will
            focus only on min_num problems
        min_ents: int or None
            the minimum number of entities within the problem. follows
            same pattern as min_num and min_mult_num
    Returns:
        probs: list or str
            a list of unique problems
    """
    if min_num is None: min_num = np.inf
    if min_mult_num is None: min_mult_num = np.inf
    if min_ents is None: min_ents = np.inf

    def get_max_mult(prob):
        splt = prob.split("*")
        if len(splt)==1: return -np.inf
        mults = []
        for i in range(len(splt)):
            m = splt[i].split("+")
            m = m[-1] if len(m)>1 else m[0]
            mults.append(int(m))
        return max(mults)

    def get_max_num(prob):
        splt = prob.split("+")
        if len(splt)==1: return -np.inf
        nums = []
        for i in range(len(splt)):
            n = splt[i]
            if "*" in n: n = max([int(x) for x in n.split("*")])
            nums.append(int(n))
        return max(nums)

    def get_ents(prob):
        chars = {"+", "*"}
        n = 0
        for p in prob:
            n += int(p in chars)
            assert p not in {"(", ")"}
        return n + 1

    probs = []
    prob_set = set()
    for i in range(n_samps):
        prob = math_env.sample()
        loop_count = 0
        while (prob in prob_set or (get_max_num(prob)<min_num and\
                get_max_mult(prob)<min_mult_num and get_ents(prob)<min_ents))\
                and loop_count<100:
            prob = math_env.sample()
            loop_count += 1
        probs.append(prob)
        prob_set.add(prob)
    return probs

def make_data_cache(probs,
                   tokenizer,
                   solns=None,
                   plen=None,
                   slen=None,
                   seq_len=None,
                   batch_size=128,
                   max_samples=None,
                   ret_strings=False,
                   incl_meta_data=False,
                   *args, **kwargs):
    """
    Creates a data_cache from the argued probs and solns

    Args:
        probs: list of str
        tokenizer: Tokenizer object
            if None, this function will create and return a tokenizer
            object
        solns: list of str or None
            solns will be generated if None is argued
        plen: int
            the length of the longest problem
        slen: int
            the length of the longest solution
        seq_len: int
            the desired sequence lengths of the samples
        batch_size: int
            the batch_size for the DataCache iterable
        max_samples: int or None
            a parameter to limit the total amount of data in the cache.
            if None, will default to init_samples
        ret_strings: bool
            if true, will return string versions of problems and
            solutions
        incl_meta_data: bool
            if true, will include the meta data in the data cache
    Returns:
        data_cache: DataCache
    """
    labels = None
    if solns is None:
        solns = []
        labels = []
        for prob in probs:
            soln, labs = envs.MathEnv.find_soln( prob, ret_labels=True )
            labels.append(tokenizer.sep.join(labs))
            solns.append( tokenizer.sep + soln )
    if plen is None:
        plen = np.max([len(p) for p in probs])
    if slen is None and seq_len is None:
        slen = np.max([len(s) for s in solns])+2 #+1 for eos, +1 for sep
    elif slen is None and seq_len:
        slen = seq_len-plen

    prob_ids = tokenizer(
        probs,
        as_tensor=True,
        max_len=plen,
        pad=True,
        add_eos=False
    )
    soln_ids = tokenizer(
        solns,
        as_tensor=True,
        max_len=slen,
        pad=True,
        add_eos=True
    )

    prob_len = prob_ids.shape[1]
    data = torch.cat([prob_ids, soln_ids], dim=1)
    if seq_len is None: seq_len = data.shape[-1]
    kwargs = {
        "max_samples": max_samples,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "init_data": data,
        "prob_len": prob_len,
    }
    if incl_meta_data:
        kwargs["meta_data"] = {
            "probs":  np.asarray(probs,  dtype="object"),
            "solns":  np.asarray(solns,  dtype="object")
        }
        if labels is not None:
            kwargs["meta_data"]["labels"] = np.asarray(
                labels,dtype="object"
            )
        kwargs["pad_id"] = tokenizer.pad_idx
    data_cache = DataCache( **kwargs )
    if ret_strings: return data_cache, probs, solns
    return data_cache

