import datas
import envs
import numpy as np
import torch

math_env = envs.ProbGen()

def tokenizer_tests():
    tokenizer = datas.get_tokenizer()
    soln = math_env.find_soln(math_env.sample_prob(20))
    toks = tokenizer(soln)
    back = tokenizer.idxs_to_strs(toks)
    assert soln==back[0]

    solns = []
    for i in range(3):
        soln = math_env.find_soln(math_env.sample_prob(20))
        solns.append(soln)
    toks = tokenizer(solns)
    back = tokenizer.idxs_to_strs(toks)
    for i in range(len(back)):
        assert back[i]==solns[i]

def data_cache_tests():
    max_len = 10
    n_samps = 5
    batch_size = 5
    tokenizer = datas.get_tokenizer()
    init_data = datas.sample_data(
        math_env, tokenizer, n_samples=n_samps, max_len=max_len
    )
    all_data = {str(x[:-1]) for x in init_data}
    cache = datas.DataCache(
        init_data=init_data, max_samples=5, seq_len=max_len, batch_size=batch_size
    )
    for samp in cache:
        for ss in samp["inputs"]:
            s = str(ss)
            assert s in all_data


if __name__=="__main__":
    #tokenizer_tests()
    data_cache_tests()
