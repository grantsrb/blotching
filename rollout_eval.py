"""
This script is used to evaluate trained models. Argue a model folder
or a path to a specific checkpoint. The results will be saved to a
csv called model_results.csv unless otherwise specified.

$ python3 multi_rollout_eval.py path/to/model_folder

Or:

$ python3 multi_rollout_eval.py path/to/model_checkpt.pt

"""

verbose = True
bsize = 1000 # Determines batch size of evaluation
overwrite = False
testing = False # printout a number of important tensors
max_num = None # override the max_num given by the hyps.
# Integer argument if you want to randomly sample n problems rather
# than systematically looking at all possible problems.
n_samples = 10000 # the number of samples. if None, does all
# uses validation data from training. takes
# priority over use_train_file
use_val_file = True # Highest priority
use_train_file = False # overwritten by use_val_file
temperature = 0.1
n_rollouts = 25
max_blotch_p = 0.4

# Env parameters
# Use None to default to the training distribution
max_num = None
max_ents = None
p_mult = None
max_mult_num = None
space_mults = None
p_ent = None

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
        splt = pred.split(sep)
        stats["pred"].append(str(sep.join(splt[1:])))
        stats["length"].append(len(pred))
        stats["resp"].append(splt[-1])
        stats["first"].append(splt[0])
    return stats


if __name__=="__main__":

    rank = 0
    if testing: print("CURRENTLY IN TESTING MODE!!!!")

    model_folders = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            if io.is_model_folder(arg):
                model_folders.append(arg)
            else:
                model_folders += io.get_model_folders(
                    arg,incl_full_path=True
                )
        elif "overwrite" in arg or "override" in arg:
            overwrite = True
        elif "train" in arg or "training" in arg:
            use_train_file = True
        elif "val" in arg or "validation" in arg:
            use_val_file = True
        elif "max_num" in arg: max_num = int(arg.split("=")[-1])
        elif "max_ents" in arg: max_ents = int(arg.split("=")[-1])
        elif "p_mult" in arg: p_mult = float(arg.split("=")[-1])
        elif "max_mult_num" in arg: max_mult_num = int(arg.split("=")[-1])
        elif "space_mults" in arg: space_mults = bool(arg.split("=")[-1])
        elif "temperature" in arg: temperature = float(arg.split("=")[-1])
        elif "n_rollouts" in arg: n_rollouts = int(arg.split("=")[-1])
        elif "max_blotch_p" in arg: max_blotch_p = float(arg.split("=")[-1])
        else:
            try:
                bsize = int(arg)
            except:
                print("Unrecognized arg", arg)

    print("Temperature:", temperature)
    print("N Rollouts:", n_rollouts)
    print("Max Blotch P:", max_blotch_p)
    if overwrite: print("Overwriting!!!")

    if use_train_file:
        results_file = "train_results.csv"
    elif use_val_file:
        results_file = "val_results.csv"
    else:
        results_file = "model_results.csv"
        if max_num is not None:
            results_file = "max_num"+str(max_num)+"_"+results_file
        if max_ents is not None:
            results_file = "max_ents"+str(max_ents)+"_"+results_file
        if p_mult is not None:
            results_file = "p_mult"+str(p_mult)+"_"+results_file
        if max_mult_num is not None:
            results_file = "max_mult_num"+str(max_mult_num)+"_"+results_file
        if space_mults is not None:
            results_file = "space_mults"+str(space_mults)+"_"+results_file
    results_file = "rollout_" + results_file

    data_caches = {}
    for f,model_folder in enumerate(model_folders):
        csv_path = os.path.join(model_folder, results_file)
        if not testing and not overwrite and os.path.exists(csv_path):
            og_df = pd.read_csv(csv_path)
            if np.any(og_df["val_temp"]==temperature):
                print(csv_path, "already exists, skipping....")
                continue
        try:
            checkpt = io.load_checkpoint(model_folder)
            hyps = checkpt["hyps"]
        except:
            continue
        print(
            "\nEvaluating", model_folder,
            "-- {}/{}".format(f,len(model_folders))
        )
        print("Temperature:", temperature)

        if "model_string" in hyps:
            hyps["model_type"] = hyps["model_string"]
        hyps["results_file"] = results_file
        hyps["seed"] = hyps.get("seed", int(time.time()))
        if hyps["seed"] is None: hyps["seed"] = int(time.time())
        torch.manual_seed(hyps["seed"])
        np.random.seed(hyps["seed"])
        hyps["loss_scale"] = 1./hyps["n_grad_loops"]
        if bsize is not None:
            hyps["batch_size"] = bsize
            hyps["val_batch_size"] = bsize
        hyps["zipf_order"] = 0 # Uniform sampling for validation

        # Establish math environment parameters
        if max_num is not None: hyps["max_num"] = max_num
        if max_ents is not None: hyps["max_ents"] = max_ents
        if p_mult is not None: hyps["p_mult"] = p_mult
        if max_mult_num is not None: hyps["max_mult_num"] = max_mult_num
        if space_mults is not None: hyps["space_mults"] = space_mults
        math_env = envs.MathEnv(**hyps)
        # Make Tokenizer
        tokenizer = datas.Tokenizer.get_tokenizer(**hyps)

        try:
            model = io.load_model(checkpt, globals())
        except:
            model = io.load_model(checkpt, globals(), load_sd=False)
            model.arange = checkpt["state_dict"]["arange"]
            model.load_state_dict(checkpt["state_dict"])
            print("state dict successfully loaded")
        model.eval()
        model.cuda()

        # Wrap model and place on gpu
        wrapped_model = LossWrapper( model, tokenizer, hyps=hyps )
        if not hyps["model_parallel"]: wrapped_model.to(rank)

        ########
        if testing: hyps["val_batch_size"] = 6
        ########

        # Make dataset
        if verbose and rank==0: print("Collecting Data")
        train_probs_file = os.path.join(model_folder, "train_probs.txt")
        with open(train_probs_file, "r") as f:
            train_probs = [p.strip() for p in f.readlines()]
        val_probs_file = os.path.join(model_folder, "val_probs.txt")
        if use_val_file and os.path.exists(val_probs_file):
            print("Using Validation Data")
            with open(val_probs_file, "r") as f:
                probs = [p.strip() for p in f.readlines()]
            if n_samples: 
                np.random.shuffle(probs)
                probs = probs[:n_samples]
            data_cache = datas.make_data_cache(
                probs,
                tokenizer,
                seq_len=hyps["seq_len"],
                batch_size=hyps["val_batch_size"],
                incl_meta_data=True,
            )
        elif use_train_file:
            print("Using Training Data")
            if n_samples: 
                np.random.shuffle(train_probs)
                train_probs = train_probs[:n_samples]
            data_cache = datas.make_data_cache(
                train_probs,
                tokenizer,
                seq_len=hyps["seq_len"],
                batch_size=hyps["val_batch_size"],
                incl_meta_data=True,
            )
        else:
            print("Using Sampled Data")
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
                    max_len=hyps["seq_len"],
                    batch_size=hyps["val_batch_size"],
                    rand_samps=n_samples,
                    held_out_probs=set(train_probs)
                )
                data_caches[cache_tup] = data_cache
        if verbose and rank==0:
            print("Total Samples:", len(data_cache))

        loss_fxn = torch.nn.CrossEntropyLoss()
        df_dict = {
            "ans": [],
            "targ": [],
            "pred_str": [],
            "prob_str": [],
            "soln_str": [],
            "label_str": [],
        }
        plen = data_cache.prob_len
        if verbose and rank==0: print("Evaluating Model")
        n_loops = len(iter(data_cache))
        wrapped_model.eval()
        model.eval()
        if hyps["model_type"]=="TransformerModel":
            bps = [0.0]
        else:
            bps = np.arange(max(model.n_btokens,1))/model.bp_gran
        bps = [bp for bp in bps if bp<max_blotch_p]
        print("Using BPs:", bps)

        ##
        if testing: bps = [0.0, 0.09]
        ##
        print("Using BPs:", bps)

        n_rolls = max(1, n_rollouts//len(bps))
        for i,data in enumerate(data_cache):
            start_time = time.time()
            if "meta_data" in data:
                meta_data = data["meta_data"]
            if not hyps["model_parallel"]:
                data["input_ids"] = data["input_ids"].to(rank)
                data["output_ids"] = data["output_ids"].to(rank)
            with torch.no_grad():
                rollout_answers = []
                for r in range(n_rolls):
                    for blotch_p in bps:
                        preds = torch.zeros(
                            data["input_ids"].shape[0],
                            len(str(math_env.get_max_val()))+3
                        ).long().to(rank) + tokenizer.pad_idx
                        print("\nBp:", blotch_p)
                        package = wrapped_model(
                            data,
                            ret_preds=True,
                            seq_len=hyps["seq_len"],
                            tforce=False,
                            prob_len=plen,
                            no_grad=True,
                            incl_all_inpts=True,
                            blotch_p=blotch_p,
                            temperature=temperature,
                        )
                        pred_ids = package["preds"]
                        pmask = utils.get_soln_mask(
                            pred_ids,
                            eos_id=tokenizer.eos_idx,
                            sep_id=tokenizer.sep_idx,
                            pad_id=tokenizer.pad_idx,
                            incl_eos=True,
                            max_len=preds.shape[-1],
                        )
                        mask = utils.mask_up_to_idx(
                            pmask.float().sum(-1), preds.shape[-1]
                        )
                        if testing:
                            print("pmask:", pmask.float().sum(-1))
                            print("mask:", mask.float().sum(-1))
                            print("unequal:", tokenizer.decode(
                                    pred_ids[pmask.float().sum(-1)==\
                                    mask.float().sum(-1)][0]
                            ))
                            out_mask = data["output_ids"]==tokenizer.pad_idx
                            in_mask = data["input_ids"]==tokenizer.pad_idx
                            for ii in range(4):
                                print(tokenizer.decode(pred_ids[ii,1:][~in_mask[ii]]))
                                print(tokenizer.decode(
                                    data["output_ids"][ii][~out_mask[ii]]
                                ))
                                print("ans:", tokenizer.decode(
                                    pred_ids[ii][pmask[ii]]
                                ))
                                print()
                        preds[mask.bool()] = pred_ids[pmask.bool()]
                        rollout_answers.append(preds)

                preds = torch.stack(rollout_answers, dim=0)
                one_hots = utils.get_one_hot( preds, model.n_tokens )
                pred_ids = torch.argmax(one_hots.float().mean(0), dim=-1)

                if testing:
                    print("preds:", preds.shape)
                    print("one_hots:", one_hots.shape)
                    print("one_hots:", one_hots[0,:4])
                    print("preds:", preds[0,:4])
                    print("argmax one_hots:", torch.argmax(one_hots,dim=-1)[0,:4])
                    print("oh mean:", one_hots.float().mean(0)[0, :4])
                    print("mean pred_ids:", pred_ids[:4])

                preds = tokenizer.decode(pred_ids)
                probs =   meta_data["probs"]
                solns =   meta_data["solns"]
                labels =  meta_data["labels"]
                for prob,soln,label,pred in zip(probs,solns,labels,preds):
                    df_dict["targ"].append(
                     soln.split(tokenizer.eos)[0].split(tokenizer.sep)[-1]
                    )
                    df_dict["soln_str"].append(soln[1:]) # removes =
                    df_dict["prob_str"].append(prob)
                    df_dict["label_str"].append(label)
                    df_dict["pred_str"].append(pred)
                    ans = pred.split(tokenizer.eos)
                    df_dict["ans"].append(ans[0])

                comps = [
                  df_dict["ans"][-j]==df_dict["targ"][-j] for j in\
                        reversed(range(1, len(preds)+1))
                ]

                correct = np.mean(comps)
                print(
                    "Correct: {} - {}% - {}s".format(
                        correct,
                        int((i+1)/n_loops*100),
                        round(time.time()-start_time, 2)
                    ),
                    end="                  \r"
                )

            ##
            if testing and i>=1:
                print("DEBUG BREAK")
                break
            ##

        print("Making pandas dataframe")
        for k in df_dict:
            print(k, "Len:", len(df_dict[k]), "- Examp:", df_dict[k][0])
        df = pd.DataFrame(df_dict)
        df["pred_str"] = df["pred_str"].astype(str)
        for k,v in hyps.items():
            try:
                df[k] = v
            except: print("error for", k)
        df["val_temp"] = temperature
        df["n_rollouts"] = n_rollouts
        df["max_blotch_p"] = max_blotch_p
        print()
        print("Avg Correct:", (df["ans"]==df["targ"]).mean())
        print("Saving...")
        if os.path.exists(csv_path) and not overwrite:
            og_df = pd.read_csv(csv_path)
            df = og_df.append(df, sort=True)
        if not testing:
            df.to_csv(csv_path, mode="w", index=False, header=True)
            print("Saved to", csv_path)
