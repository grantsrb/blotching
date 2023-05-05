"""
This script is used to evaluate trained models. Argue a model folder
or a path to a specific checkpoint. The results will be saved to a
csv called model_results.csv unless otherwise specified.

$ python3 eval_model.py path/to/model_folder

Or:

$ python3 eval_model.py path/to/model_checkpt.pt

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
import training

def get_stats(tokenizer, ids, remove_padding=True):
    stats = {
        "length": [],
        "resp": [],
        "pred": [],
        "first": [], # The chunk of string preceding the first separator
    }
    pred_strings = tokenizer.decode(ids)
    eos = tokenizer.eos
    sep = tokenizer.sep
    pad = tokenizer.pad
    for i,pred in enumerate(pred_strings):
        if remove_padding: pred = pred.replace(pad, "")
        pred = pred.split(eos)[0]
        stats["pred"].append(pred)
        stats["length"].append(len(pred))
        splt = pred.split(sep)
        stats["resp"].append(splt[-1])
        stats["first"].append(splt[0])
    return stats


if __name__=="__main__":
    rank = 0
    verbose = True
    results_file = "model_results.csv"
    abbrev_len = 1000
    bsize = None # Determines batch size of evaluation
    overwrite = False
    testing = False
    max_num = None # override the max_num given by the hyps

    if testing:
        print("CURRENTLY IN TESTING MODE!!!!")

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
    data_caches = {}
    for f,model_folder in enumerate(model_folders):
        csv_path = os.path.join(model_folder, results_file)
        if not testing and not overwrite and os.path.exists(csv_path):
            print(csv_path, "already exists, skipping....")
            continue
        print(
            "Evaluating", model_folder,
            "-- {}/{}".format(f,len(model_folders))
        )
        checkpt = io.load_checkpoint(model_folder)
        hyps = checkpt["hyps"]

        if "model_string" in hyps:
            hyps["model_type"] = hyps["model_string"]
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
        if not hyps["model_parallel"]: wrapped_model.to(rank)

        # Make dataset
        if verbose and rank==0: print("Collecting Data")
        if max_num: math_env.max_num = max_num
        elif testing: math_env.max_num = 10
        cache_tup = (
            math_env.max_num, math_env.max_ents, math_env.p_mult,
            math_env.p_paren, math_env.space_mults
        )
        if cache_tup in data_caches:
            data_cache = data_caches[cache_tup]
        else:
            data_cache = datas.get_validation_set(
                math_env,
                tokenizer,
                max_len=None,
                batch_size=hyps["val_batch_size"]
            )
            data_caches[cache_tup] = data_cache
        if verbose and rank==0:
            print("Total Samples:", len(data_cache))

        loss_fxn = torch.nn.CrossEntropyLoss()
        df_dict = {
            "tok_acc":  [],
            "ans": [],
            "targ": [],
            "pred_str": [],
            "prob_str": [],
            "soln_str": [],
            "label_str": [],
        }
        plen = data_cache.prob_len
        if verbose and rank==0: print("Evaluating")
        for i,data in tqdm(enumerate(data_cache)):
            if "meta_data" in data:
                meta_data = data["meta_data"]
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
                    prob_len=plen,
                    no_grad=True,
                    incl_all_inpts=True,
                )
                pred_ids = package["preds"]

                out_ids = data["output_ids"][:,:plen]
                probs = meta_data["probs"]
                solns =  meta_data["solns"]
                labels =  meta_data["labels"]
                for prob,soln,label in zip(probs, solns,labels):
                    df_dict["targ"].append(
                     soln.split(tokenizer.eos)[0].split(tokenizer.sep)[-1]
                    )
                    df_dict["soln_str"].append(soln[1:])
                    df_dict["prob_str"].append(prob)
                    df_dict["label_str"].append(label)

                stats = get_stats(tokenizer=tokenizer, ids=pred_ids)
                df_dict["ans"]      += stats["resp"]
                df_dict["pred_str"] += stats["pred"]

                out_ids = data["output_ids"][:,plen:]
                acc = pred_ids[:,plen+1:]==out_ids
                out_pad_mask = out_ids==tokenizer.pad_idx
                acc[out_pad_mask] = 0
                acc = acc.float().sum(-1)
                acc = acc / (~out_pad_mask).sum(-1)
                df_dict["tok_acc"].append(acc.cpu().data.numpy())

        df_dict["tok_acc"] = np.concatenate(df_dict["tok_acc"], axis=0)
        print("Making pandas dataframe")
        for k in df_dict:
            print(k, len(df_dict[k]), df_dict[k][0])
        df = pd.DataFrame(df_dict)
        for k,v in hyps.items():
            try:
                df[k] = v
            except: print("error for", k)
        print("Saving...")
        if os.path.exists(csv_path) and not overwrite:
            og_df = pd.read_csv(csv_path)
            df = og_df.append(df, sort=True)
        if not testing:
            df.to_csv(csv_path, mode="w", index=False, header=True)
            print("Saved to", csv_path)
