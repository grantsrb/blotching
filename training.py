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

    # Make Tokenizer
    max_num = hyps.get("max_num", 20)**hyps.get("max_ents", 2)
    tokenizer = datas.Tokenizer.get_tokenizer(
        digit_embs=hyps.get("digit_embs",True),
        max_num=max_num
    )
    hyps["n_tokens"] = tokenizer.n_tokens
    hyps["str2idx"] = tokenizer.str2idx

    # Make dataset
    if verbose and rank==0:
        print("Collecting Initial Data")
    if hyps["exp_name"]=="test": hyps["max_samples"] = 1000
    hyps["init_samples"] = hyps.get(
        "init_samples", hyps.get("max_samples",1000000)
    )
    data_cache = datas.get_data_cache(
        math_env,
        tokenizer,
        init_samples=hyps["init_samples"],
        seq_len=hyps["seq_len"],
        max_samples=hyps["max_samples"],
        batch_size=hyps["batch_size"],
    )
    hyps["seq_len"] = data_cache.seq_len
    print("Using Sequence Length:", hyps["seq_len"])

    model = make_model(hyps)
    hyps["model_parallel"] = hyps.get("model_parallel", False)
    if not hyps["model_parallel"]: model.to(rank)

    if hyps.get("star", False):
        collector = datas.Collector(model, hyps, tokenizer)

    if verbose and rank==0:
        print("Collecting Validation Data")
    val_samples = hyps.get("val_samples",int(0.2*hyps["max_samples"]))
    val_cache = datas.get_data_cache(
        math_env,
        tokenizer,
        init_samples=val_samples,
        seq_len=hyps["seq_len"],
        max_samples=val_samples,
        batch_size=hyps.get("val_batch_size",500),
    )

    if verbose and rank==0:
        print("Recording Session")
    if rank==0: ml_utils.training.record_session(hyps, model)

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
        if hyps.get("star",False) and epoch > hyps.get("pre_epochs", 3):
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

            avg_acc += acc.item()
            avg_loss += loss.item()

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
                    save_dict = {
                        "mid_epoch": True,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc":  train_acc,
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
        avg_loss = 0
        avg_acc = 0
        if rank==0 and epoch%val_mod==0:
            ddp_model.eval()
            if verbose:
                print("Validating...")
            with torch.no_grad():
                iterable = iter(val_cache)
                nloops = hyps.get("max_val_loops",None)
                if nloops is None: nloops = len(iterable)
                nloops = min(nloops, len(iterable))
                for i,data in enumerate(val_cache):
                    starttime = time.time()
                    if not hyps["model_parallel"]:
                        data = {k: v.to(rank) for k,v in data.items()}
                    package = ddp_model(
                        data,
                        ret_preds=True,
                        tforce=False,
                        seq_len=hyps["seq_len"],
                        prob_len=val_cache.prob_len,
                        incl_intl_prob=hyps.get("incl_intl_prob", False)
                    )
                    loss = package["loss"]
                    acc = package["acc"]
                    preds = package["preds"]

                    avg_loss += loss.item()
                    avg_acc += acc.item()
                    if hyps["exp_name"]=="test" and i>=3: break
                    if i>=nloops-l: break
                    if verbose:
                        p = round(100*i/nloops, 2)
                        t = round(time.time()-starttime, 4)
                        print("{}% -- {}s".format(p,t), end="         \r")
            div = (i+1)
            val_loss = round(avg_loss/div, 5)
            val_acc = round(avg_acc/div, 5)
            if rank==0 and verbose:
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

                s = "Train Loss: {} -Train Acc: {}".format(
                    train_loss,train_acc
                )
                print(s)
                logstr += s + "\n"
                s = "Val Loss: {} -Val Acc: {}".format(
                    val_loss,val_acc
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
            optimizer.zero_grad()
            if rank==0 and hyps.get( "save", True):
                save_dict = {
                    "mid_epoch": False,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "val_loss":   val_loss,
                    "val_acc":    val_acc,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "hyps": hyps,
                    "examples": examples,
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
        if hyps.get("star",False) and epoch > hyps.get("pre_epochs", 3):
            if rank==0 and verbose:
                print("Awaiting Runners")
            # await_harvest does nothing until collectors are dispatched
            collector.await_runners()
            new_data = collector.harvest_exp()
            data_cache.add_data(new_data)
            if rank==0 and verbose:
                print("Updating Runner Models")
            # updates the collection model with the most recent weights
            collector.update_model(model)
        keys = list(package.keys())
        for k in keys: del package[k]
        if hyps["exp_name"]=="test" and epoch==2: break
    if hyps["multi_gpu"]: dist.destroy_process_group()
    if hyps.get("star",False):
        collector.terminate_procs()
        collector.dispatch_runners()


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
