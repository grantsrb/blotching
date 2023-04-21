import datas
import envs
import numpy as np
import torch

math_env = envs.ProbGen()

def tokenizer_tests():
    tokenizer = datas.Tokenizer.get_tokenizer()
    prob = math_env.sample_prob(20)
    soln = math_env.find_soln(prob)
    print("tok test")
    print(prob)
    print(soln)
    print()
    toks = tokenizer(soln, add_eos=False)
    back = tokenizer.idxs_to_strs(toks)
    try:
        assert soln==back[0]
    except:
        print("Soln:", soln)
        print("Back:", back[0])

    solns = []
    for i in range(3):
        soln = math_env.find_soln(math_env.sample_prob(20))
        solns.append(soln)
    toks = tokenizer(solns, add_eos=False)
    back = tokenizer.idxs_to_strs(toks)
    for i in range(len(back)):
        try:
            assert back[i]==solns[i]
        except:
            print("Soln:", soln)
            print("Back:", back[0])

def data_cache_tests():
    max_len = 10
    n_samps = 5
    batch_size = 5
    tokenizer = datas.Tokenizer.get_tokenizer()
    probs, solns = datas.sample_data(
      math_env, tokenizer, n_samples=n_samps, max_len=max_len
    )
    init_data = torch.cat([probs, solns], dim=1)
    all_data = {str(x[:-1]) for x in init_data}
    cache = datas.DataCache(
        init_data=init_data, max_samples=5, seq_len=max_len, batch_size=batch_size
    )
    for samp in cache:
        for ss in samp["input_ids"]:
            s = str(ss)
            assert s in all_data


if __name__=="__main__":
    #tokenizer_tests()
    data_cache_tests()
    print("Everything seems fine?")
