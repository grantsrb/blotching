import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import time
from tqdm import tqdm
import os

import ml_utils
import datas
import models
from envs import MathEnv

RMB = "|<RMB>|" # Extra characters are to ensure uniqueness
CMP = "|<CMP{}>|"
SOS = "|<SOS>|"


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
    if hyps["p_mult"]>0:
        max_num = math_env.max_num
        max_ents = math_env.max_ents
        mmn = math_env.max_mult_num
        space = math_env.space_mults
        hyps["max_val"] = max(
          mmn**(max_ents-space*(max_ents//3))+space*(max_ents//3)*max_num,
          max_num*max_ents
        )
        print("Max Value:", hyps["max_val"])
    else:
        hyps["max_val"] = math_env.max_ents*math_env.max_num

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

    # Make dataset
    if hyps["exp_name"]=="test":
        hyps["max_samples"] = 1000
        hyps["pre_epochs"] = 0
    if verbose and rank==0:
        print("Collecting Data")
    all_problems = datas.get_all_problems(math_env, shuffle=True)
    vbsize = hyps.get("val_batch_size",500)
    if len(all_problems)<hyps["max_samples"]:
        if verbose and rank==0:
            print("Using all possible data")
        val_len = int(0.2*len(all_problems))
        val_cache, val_probs, val_solns = datas.make_data_cache(
            probs=all_problems[:val_len],
            tokenizer=tokenizer,
            solns=None,
            plen=math_env.prob_len,
            slen=math_env.max_soln_len+2,
            batch_size=vbsize,
            ret_strings=True,
        )
        data_cache = datas.make_data_cache(
            probs=all_problems[val_len:],
            tokenizer=tokenizer,
            solns=None,
            plen=math_env.prob_len,
            slen=math_env.max_soln_len+2,
            batch_size=hyps["batch_size"]
        )
        hyps["seq_len"] = val_cache.seq_len
        hyps["prob_len"] = val_cache.prob_len
    else:
        if verbose and rank==0:
            print("Using sampling process")
        val_samples = hyps.get("val_samples",int(0.2*hyps["max_samples"]))
        if hyps["exp_name"] == "test":
            val_samples = 1100
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
        if not hyps["init_samples"]:hyps["init_samples"]=hyps["max_samples"]
        data_cache = datas.get_data_cache(
            math_env=math_env,
            tokenizer=tokenizer,
            init_samples=hyps["init_samples"],
            seq_len=hyps["seq_len"],
            max_samples=hyps["max_samples"],
            batch_size=hyps["batch_size"],
            held_out_probs=val_probs
        )
    if verbose and rank==0:
        print("Train Samples:", len(data_cache))
        print("Val Samples:", len(val_cache))
        print("Using Sequence Length:", hyps["seq_len"])

    model = make_model(hyps)
    hyps["model_parallel"] = hyps.get("model_parallel", False)
    if not hyps["model_parallel"]: model.to(rank)

    if hyps.get("star", False):
        collector = datas.Collector(model, hyps, tokenizer)

    if verbose and rank==0:
        print("Recording Session")
    if rank==0:
        ml_utils.training.record_session(hyps, model)
        sf = hyps['save_folder']
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
        # If enough training, asynchronously sample new data using model
        if hyps.get("star",False) and epoch>=hyps.get("pre_epochs", 3):
            if rank==0 and verbose: print("Dispatching Runners")
            collector.dispatch_runners()

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
        optimizer.zero_grad()
        for i,data in enumerate(iterable):
            starttime = time.time()
            if not hyps["model_parallel"]:
                data = {k: v.to(rank) for k,v in data.items()}
            package = ddp_model(
                data,
                ret_preds=True,
                seq_len=hyps["seq_len"],
                tforce=True,
                prob_len=data_cache.prob_len,
                incl_intl_prob=hyps.get("incl_intl_prob", False)
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
                if hyps["model_type"]=="TransformerModel":
                    blotch_ps = [0.0]
                elif hyps["exp_name"]=="test": blotch_ps = [0]
                else:
                    blotch_ps = np.arange(max(model.n_btokens//2,1))*2
                    blotch_ps = blotch_ps/model.bp_gran
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
                            blotch_p=bp
                        )
                        loss = package["loss"]
                        acc = package["acc"]
                        len_diff = package["len_diff"]
                        len_perc = package["len_percent"]
                        preds = package["preds"]

                        corrects = check_correct(
                            tokenizer,
                            data["output_ids"],
                            preds,
                        )

                        if bp==0:
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
                        div = (i+1)
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

        ##############################################################
        #### STaR BOOTSTRAPPING
        ##############################################################
        n_new_samps = 0
        if hyps.get("star",False) and epoch>=hyps.get("pre_epochs", 3):
            if rank==0 and verbose:
                print("Awaiting Runners")
            # await_harvest does nothing until collectors are dispatched
            collector.await_runners()
            new_data = collector.harvest_exp()
            n_new_samps = len(new_data)
            if rank==0 and verbose:
                try:
                    print("New samples:", n_new_samps)
                    if n_new_samps>0:
                        print("Examples:")
                        examps = tokenizer.decode(new_data[:5])
                        for e,ex in enumerate(examps):
                            print(e,"-",ex.replace(tokenizer.pad, ""))
                except Exception as e:
                    print(e)
                    print("Issue viewing new samples")
            data_cache.add_data(new_data)
            if rank==0 and verbose:
                print("Updating Runner Models")
            # updates the collection model with the most recent weights
            collector.update_model(model)

        ##############################################################
        #### DATA AUGMENTATIONS
        ##############################################################
        n_aug_samps = 0
        if hyps.get("aug_loops",0)>0 and epoch>=hyps.get("pre_epochs",3):
            print("Augmenting Dataset")
            aug_samps = datas.augment_data(
                hyps=hyps,
                model=model,
                data_cache=data_cache,
                tokenizer=tokenizer,
                in_place=True
            )
            n_aug_samps = sum([len(x) for x in aug_samps])
            if rank==0 and verbose:
                try:
                    print("Augmented samples:", n_aug_samps)
                    if n_aug_samps>0:
                        print("Examples:")
                        examps = tokenizer.decode(aug_samps[0][:5])
                        for e,ex in enumerate(examps):
                            print(e,"-",ex.replace(tokenizer.pad, ""))
                except Exception as e:
                    print(e)
                    print("Issue viewing new samples")

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
                    "n_new_samps": n_new_samps,
                    "n_aug_samps": n_aug_samps,
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
        keys = list(package.keys())
        for k in keys: del package[k]
        if hyps["exp_name"]=="test" and epoch==2: break
    if hyps["multi_gpu"]: dist.destroy_process_group()
    if hyps.get("star",False):
        collector.terminate_procs()
        collector.dispatch_runners()

def check_correct(tokenizer, output_ids, pred_ids):
    """
    Determines whether or not the outputs are correct for each row
    of the inputs.

    Arguments:
        tokenizer: Tokenizer
        output_ids: torch LongTensor (N,S)
        pred_ids: torch LongTensor (N,T)
    Returns:
        corrects: Bool Tensor (N,)
            True values indicate that the final output of the row was
            correctly predicted.
    """
    targs = tokenizer.decode(output_ids)
    preds = tokenizer.decode(pred_ids)
    corrects = torch.zeros(len(preds))
    eos = tokenizer.eos
    sep = tokenizer.sep
    for i,(targ,pred) in enumerate(zip(targs, preds)):
        t = targ.split(eos)[0].split(sep)[-1]
        p = pred.split(eos)[0].split(sep)[-1]
        corrects[i] = t==p
    return corrects


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
        model.load_state_dict(checkpt["state_dict"])
    return model

def hyper_error_catching(hyps):
    """
    This function just makes sure that some obvious hyperparameter
    choices are set and some obviously wrong hyperparameter settings
    are changed to what the experimenter meant.
    """
    if not hyps["blotch_p"] and not hyps.get("blotch_p_min", None) and\
            not hyps.get("blotch_p_max", None):
        hyps["model_type"] = "TransformerModel"
    return hyps

