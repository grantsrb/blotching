import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import time
from tqdm import tqdm
import os

import ml_utils
import utils
import datas
import models
from envs import MathEnv

RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP{}>|"
SOS = "|<SOS>|"

DEVICES = {
    -1: "cpu",
    **{i: i for i in range(10)}
}


def train(rank, hyps, verbose=True, *args, **kwargs):
    # Distributed Set Up
    torch.cuda.empty_cache()
    hyps = hyper_error_catching(hyps)
    hyps["multi_gpu"] = hyps.get("multi_gpu", False)
    if hyps["multi_gpu"]:
        world_size = hyps.get("n_gpus", 1)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Hyperparameters
    lr = hyps["lr"]
    l2 = hyps["l2"]
    n_epochs = hyps["n_epochs"]
    hyps["seed"] = hyps.get("seed", int(time.time()))
    if hyps["seed"] is None: hyps["seed"] = int(time.time())
    torch.manual_seed(hyps["seed"]+rank)

    hyps["rank"] = rank

    # Establish math environment parameters
    math_env = MathEnv(**hyps)
    hyps["max_val"] = math_env.get_max_val()
    print("Max Value:", hyps["max_val"])

    # Make Tokenizer
    max_num = hyps.get("max_num", 20)**hyps.get("max_ents", 2)
    tokenizer = datas.Tokenizer.get_tokenizer(
        digit_embs=hyps.get("digit_embs",True),
        max_num=max_num
    )
    hyps["n_tokens"] = tokenizer.n_tokens
    hyps["str2idx"] = tokenizer.str2idx
    hyps["sep_idx"] = tokenizer.sep_idx
    hyps["pad_idx"] = tokenizer.pad_idx
    hyps["eos_idx"] = tokenizer.eos_idx

    ##################
    # Make data caches
    ##################
    # Make dataset
    if hyps["exp_name"]=="test":
        hyps["max_samples"] = 1000
        hyps["pre_epochs"] = 0
    if verbose and rank==0:
        print("Collecting Data")
    vbsize = hyps.get("val_batch_size",500)
    # if starting from some other checkpoint, we want to keep the
    # same training and validation data as before.
    if hyps.get("init_checkpt", None):
        temp_hyps = ml_utils.save_io.get_hyps(hyps["init_checkpt"])
        sf = temp_hyps['save_folder']
        with open(os.path.join(sf,"train_probs.txt"),"r") as f:
            train_probs = [p.strip() for p in f.readlines()]
        with open(os.path.join(sf,"val_probs.txt"),"r") as f:
            val_probs = [p.strip() for p in f.readlines()]
        all_problems = train_probs + val_probs
        val_len = len(val_probs)
        train_len = len(train_probs)
    else:
        # In some cases, we have fewer possible problems than our argued
        # initial training sample count. So, we first want to collect
        # all possible problems to make sure that we create an appropriate
        # data split
        all_problems = datas.get_all_problems(math_env, shuffle=True)
        # Will handle dividing all_problems or 
        val_probs = None
        train_probs = None
        val_len = min(int(0.2*len(all_problems)), hyps["val_samples"])
        train_len = len(all_problems)-val_len


    # This is the case where our total number of problems is small
    if val_probs or len(all_problems)<hyps["max_samples"]:
        if verbose and rank==0:
            print("Using all possible data")
        # Handle cases where we've already loaded the validation problsms
        if not val_probs:
            val_probs = all_problems[:val_len]
        val_cache, val_probs, val_solns = datas.make_data_cache(
            probs=val_probs,
            tokenizer=tokenizer,
            solns=None,
            plen=math_env.prob_len,
            slen=math_env.max_soln_len+2,
            batch_size=vbsize,
            ret_strings=True,
        )
        hyps["seq_len"] = val_cache.seq_len
        hyps["prob_len"] = val_cache.prob_len
        init_samps = hyps.get("init_samples", hyps["max_samples"])
        # Handle cases where we've already loaded the training problems
        if not train_probs:
            end = len(all_problems)
            if init_samps:
                end = min(val_len+init_samps, len(all_problems))
            train_probs = all_problems[val_len:end]
        else:
            end = len(train_probs)
            if init_samps:
                end = min(init_samps, end)
            train_probs = train_probs[:end]
        data_cache = datas.make_data_cache(
            probs=train_probs,
            tokenizer=tokenizer,
            solns=None,
            plen=hyps["prob_len"],
            slen=hyps["seq_len"]-hyps["prob_len"],
            seq_len=hyps["seq_len"],
            batch_size=hyps["batch_size"],
            max_samples=hyps["max_samples"],
        )
        print("Val Cache Shape:", val_cache.shape)
        print("Train Cache Shape:", data_cache.shape)
    # This is the case where we can sample freely because we have lots
    # of possible problems
    else:
        if verbose and rank==0:
            print("Using sampling process")
        val_samples = hyps.get("val_samples",int(0.2*hyps["max_samples"]))
        if hyps["exp_name"] == "test":
            val_samples = min(vbsize, val_samples)
        val_cache, val_probs, _ = datas.get_data_cache(
            math_env=math_env,
            tokenizer=tokenizer,
            init_samples=val_samples,
            seq_len=hyps["seq_len"],
            max_samples=val_samples,
            batch_size=hyps.get("val_batch_size",500),
            ret_strings=True
        )
        val_probs = set(val_probs)
        hyps["seq_len"] = val_cache.seq_len
        hyps["prob_len"] = val_cache.prob_len

        hyps["init_samples"] = hyps.get(
            "init_samples", hyps.get("max_samples",100000)
        )
        if not hyps["init_samples"]:
            hyps["init_samples"]=hyps["max_samples"]
        data_cache, train_probs, _ = datas.get_data_cache(
            math_env=math_env,
            tokenizer=tokenizer,
            init_samples=hyps["init_samples"],
            seq_len=hyps["seq_len"],
            max_samples=hyps["max_samples"],
            batch_size=hyps["batch_size"],
            held_out_probs=val_probs,
            ret_strings=True,
        )
    all_problems = set(val_probs).union(set(train_probs))
    if verbose and rank==0:
        print("Train Samples:", len(data_cache))
        print("Val Samples:", len(val_cache))
        print("Using Sequence Length:", hyps["seq_len"])

    model = make_model(hyps)
    n_params = 0
    for p in model.parameters():
        if hasattr(p, "data"):
            n_params += p.data.numel()
    hyps["n_params"] = n_params
    print("NParameters:", n_params)
    hyps["model_parallel"] = hyps.get("model_parallel", False)
    if not hyps["model_parallel"]:
        model.to(rank)

    if verbose and rank==0:
        print("Model Type:", type(model))
        print("Recording Session")
    if rank==0:
        ml_utils.training.record_session(hyps, model)
        sf = hyps['save_folder']
        with open(os.path.join(sf,"train_probs.txt"),"w") as f:
            for prob in train_probs:
                f.write(prob + "\n")
        with open(os.path.join(sf,"val_probs.txt"),"w") as f:
            for prob in val_probs:
                f.write(prob + "\n")

    # Wrap model to distribute loss calculations
    if verbose and rank==0:
        print("Wrapping Model")
    wrapped_model = models.LossWrapper( model, tokenizer, hyps=hyps )
    if not hyps["model_parallel"]:
        if verbose and rank==0:
            print("Putting Model On GPU")
        wrapped_model.to(rank)

    if verbose and rank==0:
        print("Creating Optimizer")
    optimizer = getattr(torch.optim, hyps.get("optim_type","Adam"))(
        model.parameters(),
        lr=lr,
        weight_decay=l2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, threshold=0.001, patience=hyps.get("patience",10)
    )

    if hyps["multi_gpu"]:
      if rank==0 and verbose: print("Putting model on multiple GPUs")
      ddp_model = DDP(
            wrapped_model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
      )
    else: ddp_model = wrapped_model

    #############################################################
    # Training
    #############################################################
    if rank==0 and verbose: print("Beginning Training")
    for epoch in range(n_epochs):

        epochtime = time.time()
        torch.cuda.empty_cache()
        if rank==0 and verbose:
            print("Emptying Trash")
            print()
            s = "Beginning Epoch {} - {}".format(
                epoch, hyps["save_folder"]
            )
            print(s)
            logstr = s + "\n"
        ddp_model.train()
        avg_loss = 0
        avg_acc = 0
        avg_len_diff = 0
        avg_len_perc = 0
        iterable = iter(data_cache)
        nloops = hyps.get("n_train_loops", None)
        nloops = len(iterable) if nloops is None else nloops
        nloops = min(nloops,len(iterable))
        checkpt_mod = hyps.get( "checkpt_mod", None)
        checkpt_mod = np.inf if checkpt_mod is None else checkpt_mod
        val_mod = hyps.get( "val_mod", 1)
        aug_mod = hyps.get( "aug_mod", 1)
        optimizer.zero_grad()
        for i,data in enumerate(iterable):
            starttime = time.time()
            if not hyps["model_parallel"]:
                data = {k: v.to(rank) for k,v in data.items()}
            package = ddp_model(
                data,
                ret_preds=True,
                tforce=True,
                prob_len=data_cache.prob_len,
                incl_intl_prob=hyps.get("incl_intl_prob", False),
                blotch_p=hyps.get("blotch_p", 0),
            )
            loss = package["loss"]
            acc = package["acc"]
            len_diff = package["len_diff"]
            len_perc = package["len_percent"]

            avg_acc += acc.item()
            avg_loss += loss.item()
            avg_len_diff += len_diff.item()
            avg_len_perc += len_perc.item()

            if i%hyps["n_grad_loops"]==0 or i==len(data_cache)-1:
                if hyps.get("grad_scaling",False):
                    model.embs.weight.grad.data = temp/temp.norm(2)
                if hyps.get("grad_clip",0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(), hyps["grad_clip"]
                    )
                optimizer.step()
                optimizer.zero_grad()

            if verbose and i%10==0 and rank==0:
                dec = 4
                l = round(loss.item(), dec)
                a = round(acc.item(), dec)
                c = round(100*i/nloops, 2)
                t = round(time.time()-starttime, 3)
                s = "Loss: {} -Acc: {}".format(l,a)
                s += " - {}% {}s   ".format(c,t)
                print(s, end=int(len(s)/2)*" " + "\r")
            if hyps["exp_name"]=="test" and i>=30: break
            if i>=(nloops-1): break
            if i>0 and checkpt_mod and i%checkpt_mod==0 and rank==0:
                if hyps.get( "save", True):
                    train_loss = round(avg_loss/i, 5)
                    train_acc = round(avg_acc/i, 5)
                    train_len_diff = round(avg_len_diff/i,5)
                    train_len_perc = round(avg_len_perc/i,5)
                    save_dict = {
                        "mid_epoch": True,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
                        "train_len_diff":  train_len_diff,
                        "train_len_perc":  train_len_perc,
                        "val_loss": None,
                        "val_acc":  None,
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "hyps": hyps,
                        "examples": examples,
                    }
                    ep = round(epoch+i/len(data_cache), 3)
                    ml_utils.save_io.save_checkpt(
                        save_dict=save_dict,
                        save_folder=hyps["save_folder"],
                        save_name="checkpt",
                        epoch=ep,
                        ext=".pt"
                    )
        div = (i+1)
        train_loss = round(avg_loss/div, 5)
        train_acc  = round(avg_acc/div, 5)
        train_len_diff = round(avg_len_diff/div,5)
        train_len_perc = round(avg_len_perc/div,5)
        if rank==0 and verbose:
            print()
            s = "Example Predictions On Training"
            print(s)
            logstr += s + "\n"
            inpt_dict = {
                "input_ids":  data["input_ids"],
                "output_ids": data["output_ids"],
                **package
            }
            examples,s = print_examples( inpt_dict, tokenizer )
            logstr += s + "\n"
            keys = list(inpt_dict.keys())
            for k in keys:
                inpt_dict[k] = inpt_dict[k].cpu()
            del inpt_dict
        del package["preds"]

        #############################################################
        # Validation
        #############################################################
        val_loss =     0
        val_acc =      0
        val_len_diff = 0
        val_len_perc = 0
        val_correct =  0
        val_dict = {
            "val_loss":     [], "val_acc":      [], "val_len_diff": [],
            "val_len_perc": [], "val_correct":  [], "blotch_p": [],
        }
        if rank==0 and (epoch%val_mod==0 or epoch==n_epochs-1):
            ddp_model.eval()
            if verbose:
                print("Validating...")
            with torch.no_grad():
                iterable = iter(val_cache)
                nloops = hyps.get("max_val_loops",None)
                if nloops is None: nloops = len(iterable)
                nloops = min(nloops, len(iterable))
                blotch_ps = [0.0]
                #if hyps["model_type"]=="TransformerModel":
                #    blotch_ps = [0.0]
                #elif hyps["exp_name"]=="test": blotch_ps = [0]
                #else:
                #    blotch_ps = np.arange(max(model.n_btokens//2,1))*2
                #    blotch_ps = blotch_ps/model.bp_gran
                for bp in blotch_ps:
                    print("\nBlotch P:", bp)
                    avg_loss = 0
                    avg_acc = 0
                    avg_len_diff = 0
                    avg_len_perc = 0
                    avg_correct = 0
                    for i,data in enumerate(val_cache):
                        starttime = time.time()
                        if not hyps["model_parallel"]:
                            data = {
                                k: v.to(rank) for k,v in data.items()
                            }
                        package = ddp_model(
                            data,
                            ret_preds=True,
                            tforce=False,
                            seq_len=hyps["seq_len"],
                            prob_len=val_cache.prob_len,
                            incl_intl_prob=hyps.get("incl_intl_prob", False),
                            blotch_p=bp,
                            temperature=hyps.get("temperature", None)
                        )
                        loss = package["loss"]
                        acc = package["acc"]
                        len_diff = package["len_diff"]
                        len_perc = package["len_percent"]
                        preds = package["preds"]

                        corrects = utils.vectorized_check_correct(
                            tokenizer,
                            data["output_ids"],
                            preds,
                        ).float()

                        avg_correct += corrects.mean().item()
                        avg_loss += loss.item()
                        avg_acc += acc.item()
                        avg_len_diff += len_diff.item()
                        avg_len_perc += len_perc.item()
                        if verbose:
                            p = round(100*(i+1)/nloops, 2)
                            t = round(time.time()-starttime, 4)
                            print("{}% -- {}s".format(p,t), end="         \r")
                        if i>=nloops-l: break
                    div = (i+1)
                    val_dict["val_loss"].append(round(avg_loss/div, 5))
                    val_dict["val_acc"].append(round(avg_acc/div, 5))
                    val_dict["val_len_diff"].append(round(avg_len_diff/div, 5))
                    val_dict["val_len_perc"].append(round(avg_len_perc/div, 5))
                    val_dict["val_correct"].append(round(avg_correct/div, 5))
                    val_dict["blotch_p"].append(bp)
                    if bp==0:
                        val_loss =     round(avg_loss/div, 5)
                        val_acc =      round(avg_acc/div, 5)
                        val_len_diff = round(avg_len_diff/div, 5)
                        val_len_perc = round(avg_len_perc/div, 5)
                        val_correct =  round(avg_correct/div, 5)
                    if hyps["exp_name"]=="test": break
            if verbose:
                print()
                s = "Example Predictions On Validation"
                print(s)
                logstr += s + "\n"
                inpt_dict = {
                    "input_ids": data["input_ids"],
                    "output_ids": data["output_ids"],
                    **package,
                }
                examples,s = print_examples( inpt_dict, tokenizer )
                keys = list(inpt_dict.keys())
                for k in keys: inpt_dict[k] = inpt_dict[k].cpu()
                del inpt_dict

                logstr += s + "\n"
                print()

                s = "Final Stats, Epoch: {}".format(epoch)
                print(s)
                logstr += "\n" + s + "\n"

                s = "Train Loss: {} - Train Acc: {}".format(
                    train_loss,train_acc
                )
                logstr += s + "\n"
                print(s)

                for i in range(len(val_dict["val_loss"])):
                    s = "Blotch P: {}".format(val_dict["blotch_p"][i])
                    logstr += s + "\n"
                    print(s)

                    s = "\tVal Loss: {} - Val Acc: {}".format(
                        val_dict["val_loss"][i],
                        val_dict["val_acc"][i],
                    )
                    logstr += s + "\n"
                    print(s)

                    s = "\tVal Correct: {}".format(
                        val_dict["val_correct"][i]
                    )
                    logstr += s + "\n"
                    print(s)

                    s = "\tTrain LDiff: {} - Val LDiff: {}".format(
                        train_len_diff,val_dict["val_len_diff"][i]
                    )
                    print(s)
                    logstr += s + "\n"

                    s = "\tTrain Len%: {} - Val Len%: {}".format(
                        train_len_perc,val_dict["val_len_perc"][i]
                    )
                    print(s)
                    logstr += s + "\n"


                s = "Epoch Dur: {}s".format(round(time.time()-epochtime))
                logstr += s + "\n\n\n\n"
                print(s)
                print()
                print()

            keys = list(data.keys())
            for k in keys: del data[k]

        if (epoch % aug_mod) == 0 or epoch==n_epochs-1:
            ###########################################################
            #### STaR BOOTSTRAPPING
            ###########################################################
            n_star_samps = 0
            star_loops = hyps.get("star_loops",0)
            if star_loops>0 and epoch>=hyps.get("pre_epochs",3):
                if rank==0 and verbose: print("Bootstrapping Dataset")
                new_data = datas.bootstrap_data(
                    hyps=hyps,
                    model=model,
                    tokenizer=tokenizer,
                    env=math_env,
                    held_out_probs=all_problems,
                    force_short=hyps.get("star_force_short", True),
                    verbose=rank==0 and verbose,
                )
                new_data = torch.vstack(new_data)
                data_cache.add_data(new_data)
                n_star_samps = len(new_data)

                if rank==0 and verbose:
                    try:
                        print("New samples:", n_star_samps)
                        if n_star_samps>0:
                            sf = hyps['save_folder']
                            fname = os.path.join(sf,"star_samps.txt")
                            with open(fname,"a") as f:
                                f.write("\n\nEpoch: {}\n".format(epoch))
                                print("Examples:")
                                examps = tokenizer.decode(new_data)
                                for e,ex in enumerate(examps):
                                    s = "{} - {}".format(
                                        e,ex.replace(tokenizer.pad, "")
                                    )
                                    if e < 5:
                                        print(s)
                                    f.write(s + "\n")
                    except Exception as e:
                        print(e)
                        print("Issue viewing new samples")

            ###########################################################
            #### DATA AUGMENTATIONS
            ###########################################################
            n_aug_samps = 0
            aug_loops = hyps.get("aug_loops",0)
            if aug_loops!=0 and epoch>=hyps.get("pre_epochs",3):
                print("Augmenting Dataset")
                aug_samps = datas.augment_data(
                    hyps=hyps,
                    model=model,
                    data_cache=data_cache,
                    tokenizer=tokenizer,
                    in_place=hyps["in_place"]
                )
                if not hyps["in_place"]:
                    new_data = torch.vstack(aug_samps)
                    data_cache.add_data(new_data)
                n_aug_samps = sum([len(x) for x in aug_samps])
                if rank==0 and verbose:
                    try:
                        print("Augmented samples:", n_aug_samps)
                        if n_aug_samps>0:
                            sf = hyps['save_folder']
                            fname = os.path.join(sf,"aug_samps.txt")
                            with open(fname,"a") as f:
                                f.write("\n\nEpoch: {}\n".format(epoch))
                                print("Examples:")
                                examps = tokenizer.decode(aug_samps[0])
                                for e,ex in enumerate(examps):
                                    s = "{} - {}".format(
                                        e,ex.replace(tokenizer.pad, "")
                                    )
                                    if e < 5:
                                        print(s)
                                    f.write(s + "\n")
                    except Exception as e:
                        print(e)
                        print("Issue viewing new samples")

            ###########################################################
            #### SELECTIVE AXING
            ###########################################################
            n_axed_samps = 0
            axe_loops = hyps.get("axe_loops",0)
            if axe_loops!=0 and epoch>=hyps.get("pre_epochs",3):
                print("Axing Data Samples")
                axed_samps = datas.axe_data(
                    hyps=hyps,
                    wrapped_model=wrapped_model,
                    data_cache=data_cache,
                    tokenizer=tokenizer,
                    in_place=hyps["in_place"],
                    verbose=rank==0 and verbose
                )

                if not hyps["in_place"]:
                    new_data = torch.vstack(axed_samps)
                    data_cache.add_data(new_data)
                n_axed_samps = sum([len(x) for x in axed_samps])
                if rank==0 and verbose:
                    try:
                        print("Axed samples:", n_axed_samps)
                        if n_axed_samps>0:
                            sf = hyps['save_folder']
                            fname = os.path.join(sf,"axed_samps.txt")
                            with open(fname,"a") as f:
                                f.write("\n\nEpoch: {}\n".format(epoch))
                                print("Examples:")
                                examps = tokenizer.decode(axed_samps[0])
                                for e,ex in enumerate(examps):
                                    s = "{} - {}".format(
                                        e,ex.replace(tokenizer.pad, "")
                                    )
                                    if e < 5:
                                        print(s)
                                    f.write(s + "\n")
                    except Exception as e:
                        print(e)
                        print("Issue viewing new samples")

        ##############################################################
        #### SAVING
        ##############################################################
        if rank==0 and epoch%val_mod==0:
            if hyps.get( "save", True ):
                save_dict = {
                    "mid_epoch": False,
                    "epoch":       epoch,
                    "train_loss":  train_loss,
                    "train_acc":   train_acc,
                    "train_len_diff":train_len_diff,
                    "train_len_perc":train_len_perc,
                    "val_loss":    val_loss,
                    "val_acc":     val_acc,
                    "val_len_diff":val_len_diff,
                    "val_len_perc":val_len_perc,
                    "val_correct": val_correct,
                    "val_dict": val_dict,
                    "n_star_samps": n_star_samps,
                    "n_aug_samps": n_aug_samps,
                    "n_axed_samps": n_axed_samps,
                    "state_dict":  model.state_dict(),
                    "optim_dict":  optimizer.state_dict(),
                    "hyps":        hyps,
                    "examples":    examples,
                }
                ml_utils.save_io.save_checkpt(
                    save_dict=save_dict,
                    save_folder=hyps["save_folder"],
                    save_name="checkpt",
                    epoch=epoch,
                    ext=".pt"
                )
                save_training_log(hyps, logstr)
            scheduler.step(val_loss)

        if rank==0 and verbose:
            print("Total Training Samples:", len(data_cache))
        keys = list(package.keys())
        for k in keys: del package[k]
        if hyps["exp_name"]=="test" and epoch>2: break
    if hyps["multi_gpu"]: dist.destroy_process_group()


def print_examples(inpt_dict, tokenizer, n_samps=5):
    """
    Helper function to print the model's predictions

    Args:
        inpt_dict: dict {str: tensor (B,Sn)}
            input_ids: torch tensor (B,S1)
                the ground truth of the compressed context ids
            output_ids: torch tensor (B,S2)
                the target ids
            pred: torch tensor (B,S1,L)
                the predicted compressed context logits
        tokenizer: huggingface tokenizer
        n_samps: int
            the number of samples to print and collect
    Returns:
        examples: list of dicts of str
            a list of the printed examples. the dicts have keys of
            "targs" and "preds"
        logstr: str
            a single string of one printout loop
    """
    tensors = []
    targs = inpt_dict["input_ids"]
    tensors.append(targs)
    preds = dict()
    for k in ["preds"]:
        if len(inpt_dict[k].shape)==3:
            preds[k] = inpt_dict[k].argmax(-1)
        else: preds[k] = inpt_dict[k]
    tensors = tensors + [v for v in preds.values()]

    lens = []
    l = min([len(t) for t in tensors])
    logstr = ""
    examples = []
    for i in range(min(n_samps, l)):
        examp = {}
        print("Samp", i)

        targ = tokenizer.decode(targs[i])[0]
        targ = targ.replace("\n","\\n")
        s = "Pad Sample: " +  targ
        if i == 0:
            logstr += s + "\n"
        print(s)

        targ = targ.replace(tokenizer.pad, "")
        s = "Sample: " +  targ
        if i == 0:
            logstr += s + "\n"
        print(s)
        examp["targ"] = targ

        mask = ~(inpt_dict["output_ids"][i]==tokenizer.pad_idx)
        for k,v in preds.items():
            eos_idx = torch.argmax(
                (v[i]==tokenizer.eos_idx).long(),
                dim=0
            ).item()
            if eos_idx <= 0: eos_idx = len(v[i])

            trunc_v = v[i][:eos_idx+1]
            pred = tokenizer.decode( trunc_v )[0].replace("\n", "\\n")
            s = k + ": " + pred.replace(tokenizer.pad, "")
            if i == 0:
                logstr += s + "\n"
            print(s)

            trunc_v = v[i][mask]
            pred = tokenizer.decode( trunc_v )[0].replace("\n", "\\n")
            s = "MaskPad"+k + ": " + pred
            if i == 0:
                logstr += s + "\n"
            print(s)

            examp[k] = pred
        print()
        examples.append(examp)
    return examples, logstr

def save_training_log(hyps, logstr, fname="training_log.txt", reset=False):
    """
    Saves the logstr to the save folder under the name training_log.txt

    hyps: dict
    logstr: str
        the string to save
    fname: str
        the name of the file to save to
    reset: bool
        if true, resets the training log and then writes. otherwise
        appends to training log
    """
    mode = "w" if reset else "a"
    with open(os.path.join(hyps["save_folder"], fname),mode) as f:
        f.write(logstr)

def make_model(hyps):
    """
    Makes the model. The model type specified in the hyperparams must
    be imported into the global scope.

    Args:
        hyps: dict
            dict of hyperparameters. See `README.md` for details
    """
    if hyps.get("max_posencs", None) is None:
        hyps["max_posencs"] = hyps["seq_len"]*3
    model = models.__dict__[hyps["model_type"]](**hyps)
    init_checkpt = hyps.get( "init_checkpt", None)
    if init_checkpt is not None and init_checkpt.strip()!="":
        print("Initializing from checkpoint", init_checkpt)
        checkpt = ml_utils.save_io.load_checkpoint(init_checkpt)
        try:
            model.load_state_dict(checkpt["state_dict"])
        except:
            model.arange = checkpt["state_dict"]["arange"]
            model.load_state_dict(checkpt["state_dict"])
    return model

def hyper_error_catching(hyps):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    if not hyps.get("blotch_p_min", None) and\
            not hyps.get("blotch_p_max", None):
        if hyps["model_type"]=="HFBlotchModel":
            hyps["model_type"] = "HFModel"
        else:
            hyps["model_type"] = "TransformerModel"
    if "init_data" in hyps:
        print("assuming init_data was meant to be init_samples")
        hyps["init_samples"] = hyps["init_data"]
    if hyps["init_samples"] < hyps["batch_size"]:
        hyps["batch_size"] = hyps["init_samples"]
    return hyps

