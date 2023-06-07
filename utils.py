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

def vectorized_check_correct(
        tokenizer,
        output_ids,
        pred_ids,
        has_conf=False
    ):
    """
    Determines whether or not the outputs are correct for each row
    of the inputs in a semi-vectorized way.

    Finds last separator and eos to determine window of answer.
    Then uses sum of matching tokens to determine if correct.

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
    eos = tokenizer.eos_idx
    sep = tokenizer.sep_idx
    device = pred_ids.get_device()

    pred_ids[:,-1] = eos
    seps = output_ids==sep
    aranges = torch.arange(pred_ids.shape[1])[None].repeat(
        (len(pred_ids), 1)
    ).to(device)
    # We find the eos in both predictions and solutions
    # We then find the ground truth solution length
    # We then create a mask that spreads the length of the soln starting
    #   from the eos token
    # Finally we compare the masked preds and masked solns
    pred_ends = torch.argmax( (pred_ids==eos).long(), dim=-1 )
    soln_ends = torch.argmax( (output_ids==eos).long(),dim=-1 )
    last_sep_idxs = torch.argsort(
        seps.long(), dim=-1, descending=True
    )[torch.arange(len(seps)).long(), seps.sum(-1).long()-1]+int(has_conf)
    soln_ends = soln_ends[:,None]
    soln_lens = soln_ends - last_sep_idxs[:,None]
    soln_idxs =(aranges<soln_ends)&(aranges>=(soln_ends-soln_lens))
    ans_idxs = (aranges<pred_ends[:,None])&\
               (aranges>=(pred_ends[:,None]-soln_lens))

    corrects = torch.zeros_like(output_ids).float()
    idx = pred_ids[ans_idxs]==output_ids[soln_idxs]
    corrects[soln_idxs] = (idx).float()
    corrects = corrects.sum(-1)
    corrects = corrects.squeeze()==soln_lens.squeeze()
    return corrects

def get_blotch_mask(
        idxs,
        sep_idx,
        blotch_p=0.4,
        allow_contig=True,
        indy_blotching=False,
        step_idx=None,
    ):
    """
    Blotches out entire segments of the input based on the sep_idx. For
    example, if you had the sequence

        [ sep, id1, id2, id3, sep, id4, id5, sep, ... ]

    then a valid blotching would be a binary mask over any sequence
    ending in sep. So, one valid blotching in this case would be

        [ 0, 1, 1, 1, 1, 0, 0, 0, ... ]

    Another valid blotching would be:

        [ 0, 0, 0, 0, 0, 1, 1, 1, ... ]

    Args:
        idxs: torch Tensor (batch_size, seq_len)
            a batch of indices
        sep_idx: int
            the separation index. in this project, the separation index
            is most likely the equals sign.
        blotch_p: float or torch float tensor (B,1)
            a tensor of the probability of blotching a valid, blotchable
            sequence for each data sample.
        allow_contig: bool 
            if true, will allow contiguous blotches. If false, will
            separate blotch segments by at least one semantic step
        indy_blotching: bool
            one can make the mask such that the same blotching pattern
            is used for all tokens in a sequence, or blotching patterns
            can be unique to each token. If this value is true, the
            function will return a mask of shape (batch_size, seq_len,
            seq_len). Otherwise it will return a mask of shape
            (batch_size, seq_len)
        step_idx: None or int or LongTensor (batch_size,)
            if int is argued, will only drop the argued semantic step.
            An argument of 0 refers the first semantic step following
            the initial problem. This function will not blotch the
            final answer.
    Returns:
        blotch_mask: torch BoolTensor (B,S)
            the shape will depend on the argument for indy_blotching
            but indy_blotching is not yet implemented.
    """
    seps = (idxs==sep_idx)
    mask = torch.zeros_like(idxs).bool()

    if indy_blotching:
        # Need to individually blotch along each sequence
        raise NotImplemented
    elif step_idx is not None:
        seps = seps.long()
        arange = torch.arange(seps.shape[0]).long()
        args = torch.argsort(-seps, dim=-1)
        if type(step_idx)==type(int()):
            max_idx = step_idx
            step_idx = torch.zeros_like(arange)+step_idx
        else:
            max_idx = torch.max(step_idx)
        for i in range(max_idx):
            tempx = i<step_idx
            seps[arange[tempx], args[tempx,i]] = 0
        sep_coords = torch.nonzero(seps)
        prev_row = -1
        for i in range(len(sep_coords)-1):
            # Check to blotch and both sep_coords are on same row
            row,col = sep_coords[i]
            # The key is prev_row!=row to only do a single blotch step
            if sep_coords[i+1][0]==row and prev_row!=row:
                _,stop_col = sep_coords[i+1]
                mask[row, col+1:stop_col+1] = True
                is_contig = True
            prev_row = row
    else:
        sep_coords = torch.nonzero(seps)
        if type(blotch_p)==type(float()):
            blotch_p = torch.full((1,), blotch_p)
        elif len(blotch_p.shape)==1: blotch_p = blotch_p[..., None].cpu()
        do_blotches = torch.rand(idxs.shape)<=blotch_p
        is_contig = False
        for i in range(len(sep_coords)-1):
            # Check to blotch and both sep_coords are on same row
            row,col = sep_coords[i]
            if sep_coords[i+1][0]==row and do_blotches[row,col]:
                if allow_contig or not is_contig:
                    _,stop_col = sep_coords[i+1]
                    mask[row, col+1:stop_col+1] = True
                    is_contig = True
            else: is_contig = False
    return mask

if __name__=="__main__":
    pred = torch.LongTensor( [0,1,2,3,4,5,6,7] )
    targ = torch.LongTensor( [2,1,0,3,4,5,6,7] )

