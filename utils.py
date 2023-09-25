import torch
import math

DEVICES = {
    **{-1: "cpu"},
    **{i:i for i in range(10)}
}

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
        output_ids: torch LongTensor (M,S)
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
    pad = tokenizer.pad_idx
    device = DEVICES[pred_ids.get_device()]

    pred_ids = pred_ids.clone()
    pred_ids[:,-1] = eos
    pseps = pred_ids==sep
    oseps = output_ids==sep
    oaranges = torch.arange(output_ids.shape[1])[None].repeat(
        (len(output_ids), 1)
    ).to(device)
    paranges = torch.arange(pred_ids.shape[1])[None].repeat(
        (len(pred_ids), 1)
    ).to(device)
    # We find the eos in both predictions and solutions
    # We then find the ground truth solution length
    # We then create a mask that spreads the length of the soln starting
    #   from the eos token
    # Finally we compare the masked preds and masked solns
    pred_ends = torch.argmax( (pred_ids==eos).long(), dim=-1 )
    soln_ends = torch.argmax( (output_ids==eos).long(),dim=-1 )

    oseps = (oseps|((oaranges<soln_ends[:,None])&(output_ids==pad))).long()

    last_sep_idxs = arglast(oseps.long(), dim=-1) + int(has_conf)
    soln_ends = soln_ends[:,None]
    soln_lens = soln_ends - last_sep_idxs[:,None]
    soln_idxs = (oaranges<soln_ends)&(oaranges>=(soln_ends-soln_lens))

    pred_ends = pred_ends[:,None]
    ans_idxs = (paranges<pred_ends)&(paranges>=(pred_ends-soln_lens))

    corrects = torch.zeros_like(pred_ids).float()
    try:
        idx = pred_ids[ans_idxs]==output_ids[soln_idxs]
    except:
        print("Error occurred in vectorized_check_correct")
        differs = ans_idxs.float().sum(-1)!=soln_idxs.float().sum(-1)
        preds = pred_ids[differs]
        solns = output_ids[differs]
        for j,(p,s) in enumerate(zip(preds,solns)):
            print("Loop:", j)
            print("Pred:", tokenizer.decode(p))
            print("Soln:", tokenizer.decode(s))
            print()
        corrects = check_correct(tokenizer,output_ids,pred_ids,has_conf)
        assert False
        return corrects.to(device)
    corrects[ans_idxs] = (idx).float()
    corrects = corrects.sum(-1)
    corrects = corrects.squeeze()==soln_lens.squeeze()
    return corrects

def get_soln_mask(
        ids,
        eos_id,
        sep_id,
        pad_id,
        has_conf=False,
        incl_eos=False,
        max_len=None,
    ):
    """
    Finds a boolean mask over the final answer, no sep or eos included.
    eos can be included if incl_eos is true.

    # We find the eos in solutions
    # We then find the ground truth solution length
    # We then create a mask that spreads the length of the soln starting
    #   from the eos token

    Arguments:
        ids: torch LongTensor (N,T)
        eos_id: int
        sep_id: int
        pad_id: int
        has_conf: bool
            if true, will assume a confidence prediction follows the sep
            token
        incl_eos: bool
            if true, will include the eos token in the mask
        max_len: int or None
            the maximum length response. if None, no maximum
    Returns:
        ans_mask: Bool Tensor (N,)
            True values indicate the final output. excludes the separator.
            includes the eos if incl_eos is true.
    """
    if max_len is None: max_len = ids.shape[-1]
    device = DEVICES[ids.get_device()]
    ids = ids.clone()
    ids[:,-1] = eos_id

    aranges = torch.arange(ids.shape[1])[None].repeat(
        (len(ids), 1)
    ).to(device)

    soln_ends = torch.argmax( (ids==eos_id).long(),dim=-1 )[:,None]
    ids[aranges>soln_ends] = pad_id

    # Need to look at padding because it's possible to have
    # =PPPP45E as an answer. Padding is not to be worried about later
    # because it's factored into the sum to find the last index
    seps = ids==sep_id
    seps = (seps|((aranges<soln_ends)&(ids==pad_id)))

    last_sep_idxs = arglast(seps.long(), dim=-1)+int(has_conf)
    # Subtract 1 to exclude the separator
    soln_lens = soln_ends - last_sep_idxs[:,None] - 1
    soln_lens[soln_lens>=max_len] = max_len-int(incl_eos)
    if incl_eos:
        soln_idxs = (aranges<=soln_ends)&(aranges>=(soln_ends-soln_lens))
    else:
        soln_idxs = (aranges<soln_ends)&(aranges>=(soln_ends-soln_lens))
    return soln_idxs

def arglast(mask, dim=-1):
    """
    This function finds the index of the last max value along a given
    dimension. torch.flip creates a copy of the tensor, so it's
    actually not as efficient as using numpy's np.flip which only
    returns a view.

    Args:
        mask: bool (B,N)
        dim: int
    Returns:
        the index of the last true value along the dimension
    """
    argmaxs = torch.argmax(torch.flip(mask, dims=(dim,)), dim=dim)
    return mask.shape[dim] - argmaxs - 1

def mask_up_to_idx(idxs, M, from_left=True):
    """
    Args:
        idxs: torch long tensor (B,)
        from_left: bool
    Returns:
        mask: torch bool tensor (B, M)
    """
    B = idxs.shape[0]
    device = idxs.get_device()
    if device<0: device = "cpu"
    arange = torch.arange(M).to(device)[None].repeat((B,1))
    if from_left:
        mask = arange<idxs[:,None]
    else:
        mask = arange>=idxs[:,None]
    return mask

def get_blotch_mask(
        idxs,
        sep_idx,
        blotch_p=0.4,
        allow_contig=True,
        indy_blotching=False,
        step_idx=None,
        spacing="random",
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
            final answer or the initial problem. The final answer is
            defined by the content after the last sep token and
            including the eos token.
        spacing: str
            an argument to decide how the blotching will be spaced.
            possible arguments are the following:
                "random": blotching is random according to the blotching
                    probability
                "equal": the blotching is semi-deterministicly
                    distributed relatively equally amongst the possible
                    segments.  The number of blotched segments is decided
                    by `blotch_p * <num possible segments>` where the
                    rounding direction is decided randomly using a
                    probability equal to the remaining fraction. So,
                    there's still some stochasticity but it's lower
                    variance and deterministic in many cases.

                TODO (DOES NOT EXIST):
                  "random_equal": the blotching is mostly spaced equally
                    except that each equally spaced blotch segment has
                    an equal probability of being located at the segment
                    before or after the equally spaced segement. i.e.
                    there is 2/3 probability that instead of blotching
                    the segment chosen by equal spacing, we instead
                    blotch the segment immediately before or after with
                    equal probability.
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
        if type(blotch_p)==type(float()) or type(blotch_p)==type(int()):
            blotch_p = torch.full((1,1), blotch_p)
        elif len(blotch_p.shape)==1: blotch_p = blotch_p[..., None].cpu()

        if spacing=="random": 
            do_blotches = torch.rand(idxs.shape)<=blotch_p
        else:
            do_blotches = get_equal_spaced_do_blotches(
                idxs, blotch_p, seps
            )

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

def get_equal_spaced_do_blotches(idxs, blotch_p, seps):
    # Complicated indexing to evenly space out the blotching steps
    # The indexing first calculates the number of blotches based
    # on the total possible and the argued proportion, blotch_p.
    # Then it determines the number of indices to skip for each
    # blotching segment. Lastly, it indexes into the ordered
    # separator indices to mark which segments should be blotched.
    device = idxs.get_device()
    if device < 0: device = "cpu"
    n_poss_blotches = seps.float().sum(-1) - 1
    n_blotch = blotch_p.squeeze().to(device)*n_poss_blotches
    n_blotches = torch.round(n_blotch).float()

    # Randomly add another blotch with probability equal to the
    # remainder
    #flr = torch.floor(n_blotch)
    #extra = ((n_blotch-flr)>torch.rand_like(n_blotch)).long()
    #extra = torch.round((n_blotch-flr))
    #n_blotches = flr + extra
    #n_blotches = n_blotches.float()

    # Find edges of bins in terms of indexes into the separators
    # Index steps which will be used to index into the sep
    # indices
    idx_space = n_poss_blotches/n_blotches
    inf_idx = n_blotches<=0 # only care about ==0, <= is better for floats
    idx_space[inf_idx] = -1
    m = int(torch.max(n_blotches))
    steps = [ idx_space*i for i in range(m-1) ]
    steps = torch.vstack(steps).T
    offsets = (n_poss_blotches - idx_space*(n_blotches-1))/2
    steps = torch.round(steps + torch.round(offsets[:,None])).long()
    steps = steps + inf_idx[:,None].long()*-seps.shape[-1]

    # Arguments where separators come ordered first
    args = torch.arange(seps.shape[-1]).repeat((seps.shape[0],1))
    args[~seps] = seps.shape[-1]
    args = torch.argsort(args,dim=-1).to(device)
    do_blotches = torch.zeros_like(idxs).bool()

    # Will index into do_blotches at sep locations and set
    # to 1 in cases 
    row_idx = torch.arange(len(do_blotches)).long().to(device)
    for s in range(steps.shape[1]):
        select = steps[:,s]>=0
        rows = row_idx[select]
        step = steps[select,s]
        do_blotches[rows, args[rows, step]] = True
    return do_blotches

def get_tok_mask(src, tok_p, sep_idx):
    """
    Returns a mask to randomly excise tokens from the context. Only
    drops tokens within the possible blotching locations.

    Args:
        src: torch Long tensor (B, S)
        tok_p: float or Float Tensor (B,S)
            the probability of dropping a token. 1 means all tokens
            are dropped. Only drops tokens within the possible
            blotching locations.
        sep_idx: int
            the separation index. in this project, the separation index
            is most likely the equals sign.
    Returns:
        mask: torch bool tensor (B,S)
    """
    temp = get_blotch_mask(
        src,
        sep_idx=sep_idx,
        blotch_p=1.,
    )
    tok_mask = torch.rand(src.shape)<tok_p
    tok_mask[:,0] = False
    tok_mask[:,-1] = False
    tok_mask = tok_mask.to(DEVICES[src.get_device()])
    return tok_mask&temp


def get_pos_ids(mask, arange=None, pad_pos_skip=True):
    """
    Returns the position ids based off of the true values of the
    mask. For example:

        Mask:     [0,1,0,1,1,0,0,1]
        Pos Ids:  [0,0,0,1,2,0,0,3]

    Args:
        mask: bool tensor (B,S)
            true means this function will find position id for it.
        arange: torch long tensor (S,) or longer
            a torch arange tensor spanning at least the length of the
            sequence
        pad_pos_skip: bool
            if false, will simply return arange
    Returns:
        pos_ids: long tensor (B,S)
    """
    device = DEVICES[mask.get_device()]
    if mask is None: return None
    B,S = mask.shape
    if arange is None:
        arange = torch.arange(S).long().to(device)
    rep = arange[:S][None].repeat((B,1)).long()

    if not pad_pos_skip: return rep

    pos_ids = torch.zeros((B,S), device=device).long()
    mask_sums = mask.long().sum(-1)

    pos_ids[mask.bool()] = rep[rep<mask_sums[:,None]]
    return pos_ids

def excise_tokens(src, mask, pad_id=0, pad_left=False):
    """
    Will create a new tensor that makes all masked ids along
    the last dimension contiguous.

    Args:
        src: torch long tensor  (B, S)
        mask: torch bool tensor (B, S)
            will excise tokens where mask is 0. keeps tokens where mask
            is 1
        pad_left: bool
            will determine whether to pad the final contiguous outputs
            on the left or on the right
    Returns:
        contig: torch long tensor (B S)
    """
    device = mask.get_device()
    if device<0: device = "cpu"
    B,S = src.shape
    contig = torch.zeros_like(src) + pad_id
    if pad_left:
        cmask = torch.arange(S-1, -1, -1)
    else:
        cmask = torch.arange(S)
    cmask = cmask[None].repeat((B,1)).long().to(device)
    cmask = cmask<mask.float().sum(-1)[..., None]
    contig[cmask] = src[mask.bool()]
    return contig, ~cmask

def reverse_excision(logits, cur_mask, old_mask):
    """
    Reverses the excision operation by placing the logits back where
    they were before. It does not place the excised tokens back in
    place though. 

    Args:
        logits: torch tensor (B,S,L)
        cur_mask: torch bool tensor (B,S)
            the current pad mask where 1 means padding
        old_mask: torch bool tensor (B,S)
            the old pad mask where 1 means padding
    Returns:
        logits: torch tensor (B,S,L)
            the rearranged logits based on the old mask
    """
    B,S = old_mask.shape
    og_logits = torch.zeros_like(logits)
    og_logits[~old_mask.bool()] = logits[~cur_mask.bool()]
    return og_logits

def get_one_hot(ids, L):
    """
    Args:
        ids: torch long tensor (..., N)
        L: int
            the length of the one-hot vector
    Returns:
        one_hots: torch long tensor (..., N, L)
    """
    shape = [*ids.shape, L]
    device = ids.get_device()
    if device<0: device = "cpu"
    one_hots = torch.zeros( shape, device=device )
    one_hots.scatter_(
        dim=-1,
        index=ids[...,None],
        src=torch.ones_like(one_hots)
    )
    return one_hots

def print_tensor(tensor):
    s = ""
    for row in range(tensor.shape[0]):
         s += "\t" + str( tensor[row].tolist() ) + "\n"
    return s

def int_linspace(low, high, n_steps):
    """
    Want a list of integers that bin the distance from low to high
    into `n_steps` equal sized bins. This function returns a list
    of length `n_steps-1` with integers denoting the edges of each
    bin.
    """
    diff = high-low
    space = diff/n_steps
    edges = torch.FloatTensor([space*i for i in range(n_steps-1)])
    offset = (diff - edges[-1])/2
    return (edges + offset).long()

if __name__=="__main__":
    print("linspace:", int_linspace(0,5,3))
    sep_id = -1
    seq_len = 25
    n_rows = 5
    blotch_p = torch.FloatTensor([0.5 for _ in range(n_rows)])
    #blotch_p = 0.7
    src = torch.arange(seq_len)[None].repeat((n_rows,1))
    for i in range(len(src)):
        src[i,i::i+2] = sep_id
    print("Blotch P:", blotch_p)
    print("Src:", print_tensor(src))
    #print("SepCounts:", (src==sep_id).long().sum(-1))
    mask = get_blotch_mask(src, sep_idx=sep_id, blotch_p=blotch_p, spacing="equal")
    seps = src==sep_id
    print("Seps:", print_tensor((seps).long()))
    print("Mask:", print_tensor(mask.long()))
    mask = mask.long()
    mask[seps] = 5
    print("SMaks:", print_tensor(mask.long()))
    #print("Mask  :", mask)
    #contig,new_mask = excise_tokens(src, ~mask.bool(), pad_left=True)

    #print("Contig:", contig)
    #rev = reverse_excision(contig, new_mask, mask.bool())
    #print("Revers:", rev)
