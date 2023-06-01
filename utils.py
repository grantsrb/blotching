import torch

def check_correct(tokenizer, output_ids, pred_ids, has_conf=False):
    """
    Determines whether or not the outputs are correct for each row
    of the inputs.

    Arguments:
        tokenizer: Tokenizer
        output_ids: torch LongTensor (N,S)
        pred_ids: torch LongTensor (N,T)
        has_conf: bool
            if true, will assume a conf prediction follows the sep
            token
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
        if has_conf:
            t = t[1:]
            p = p[1:]
        corrects[i] = t==p
    return corrects

