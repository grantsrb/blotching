"""
This script is used to evaluate trained models. Argue a model folder
or a path to a specific checkpoint. The results will be saved to a
csv called model_results.csv unless otherwise specified.

$ python3 eval_model.py path/to/model_folder

Or:

$ python3 eval_model.py path/to/model_checkpt.pt

If you would like to run the untrained model to see the baseline
performance, either use the `baseline_performance.py` script or
include `untrained` in the bash command.

WARNING!!! THE FOLLOWING LINE IS FOR BASELINE RESULTS:
$ python3 eval_model.py path/to/model_folder untrained
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os

import ml_utils.save_io as io
import datas
from models import *
import envs

if __name__=="__main__":
    rank = 0
    verbose = True
    results_file = "model_results.csv"
    abbrev_len = 1000
    bsize = None # Determines batch size of evaluation
    overwrite = False

    model_folders = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            if io.is_model_folder(arg):
                model_folders.append(arg)
            else:
                model_folders += io.get_model_folders(
                    arg,incl_full_path=True
                )
        elif "overwrite" in arg:
            overwrite = True
        else:
            try:
                bsize = int(arg)
            except:
                print("Unrecognized arg", arg)
    if overwrite: print("Overwriting!!!")
    for f,model_folder in enumerate(model_folders):
        csv_path = os.path.join(model_folder, results_file)
        if not overwrite and os.path.exists(csv_path):
            print(csv_path, "already exists, skipping....")
            continue
        print(
            "Evaluating", model_folder,
            "-- {}/{}".format(f,len(model_folders))
        )
        checkpt = io.load_checkpoint(model_folder)
        hyps = checkpt["hyps"]

        hyps["model_type"] = hyps["model_string"]
        if abbrev_len is not None: hyps["abbrev_len"] = abbrev_len
        hyps["results_file"] = results_file
        hyps["seed"] = hyps.get("seed", int(time.time()))
        if hyps["seed"] is None: hyps["seed"] = int(time.time())
        torch.manual_seed(hyps["seed"])
        hyps["loss_scale"] = 1./hyps["n_grad_loops"]
        if bsize is not None:
            hyps["batch_size"] = bsize
            hyps["val_batch_size"] = bsize

        # Establish math environment parameters
        math_env = envs.MathEnv(**hyps)

        # Make Tokenizer
        tokenizer = datas.Tokenizer.get_tokenizer(**hyps)

        model = io.load_model(checkpt, globals())
        model.eval()
        model.cuda()

        # Wrap model and place on gpu
        wrapped_model = LossWrapper( model, tokenizer, hyps=hyps )
        if not hyps["model_parallel"]:
            wrapped_model.to(rank)

        # Make dataset
        if verbose and rank==0: print("Collecting Data")
        data_cache = datas.get_validation_set(
            math_env,
            tokenizer,
            max_len=None,
            batch_size=hyps["val_batch_size"]
        )
        if verbose and rank==0: print("Total Samples:", len(data_cache))

        loss_fxn = torch.nn.CrossEntropyLoss()
        keys = math_env.sample_sequence()[-1].keys()
        df_dict = {
            **{ "tok_acc":  [], "n_resps":  []},
            **{k: [] for k in keys}
        }
        plen = data_cache.prob_len
        if verbose and rank==0: print("Evaluating")
        for i,data in tqdm(enumerate(data_cache)):
            meta_data = data["meta_data"]
            for k in meta_data:
                df_dict[k].append(meta_data[k])
            if not hyps["model_parallel"]:
                data["input_ids"] = data["input_ids"].to(rank)
                data["output_ids"] = data["output_ids"].to(rank)
            with torch.no_grad():
                wrapped_model.eval()
                model.eval()

                package = wrapped_model(
                    data,
                    ret_preds=True,
                    seq_len=hyps["seq_len"],
                    tforce=False,
                    prob_len=data_cache.prob_len,
                    no_grad=True
                )
                preds = package["preds"][:,plen:]

                counts = envs.CountEnv.parse_counts(
                    preds,
                    resp_only=True,
                ) + int(not hyps.get("incl_bos", True))
                df_dict["n_resps"].append(counts.cpu().data.numpy())

                out_ids = data["output_ids"][:,plen:]
                acc = preds==out_ids
                out_pad_mask = out_ids==tokenizer.pad_idx
                acc[out_pad_mask] = 0
                acc = acc.sum(-1)
                acc = acc / (~out_pad_mask).sum(-1)
                df_dict["tok_acc"].append(acc.cpu().data.numpy())

                good_idxs = counts.cpu()==meta_data["n_targs"].cpu()
                if df_dict["tok_acc"][-1].mean() > 0.9 and \
                              (good_idxs).float().mean()<0.9:

                    print()
                    print("NTarg:", meta_data["n_targs"][~good_idxs][0])
                    print("Count:", counts[~good_idxs][0])
                    print("Inpt ids:", data["input_ids"][~good_idxs][0])
                    print("Targ ids:", data["output_ids"][~good_idxs][0])
                    print("Preds:", package["preds"][~good_idxs][0])
                    print("Abrv Targs:", out_ids[~good_idxs][0])
                    print("Preds:", preds[~good_idxs][0])
                    #inpts = data["input_ids"][:,plen:]
                    #pad_mask = (inpts==tokenizer.pad_idx)|\
                    #            (inpts==tokenizer.eos_idx)
                    #print("Out Pad:", out_pad_mask[0])
                    #print("In Pad:", pad_mask[0])
                    #assert np.array_equal(
                    #    out_pad_mask.cpu().numpy(), pad_mask.cpu().numpy()
                    #)

        for k in df_dict:
            df_dict[k] = np.concatenate(df_dict[k], axis=0)
        df = pd.DataFrame(df_dict)
        print("Making pandas dataframe")
        for k,v in hyps.items():
            try:
                df[k] = v
            except: print("error for", k)
        print("Saving...")
        if os.path.exists(csv_path) and not overwrite:
            og_df = pd.read_csv(csv_path)
            df = og_df.append(df, sort=True)
        df.to_csv(csv_path, mode="w", index=False, header=True)
        print("Saved to", csv_path)

