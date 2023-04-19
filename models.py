from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
import ml_utils.utils as utils
import math

class Model(torch.nn.Module):
    def __init__(self, n_tokens: int,
                       d_model:int=128,
                       n_heads:int=4,
                       h_mult:int=4,
                       n_layers:int=3,
                       posenc_type:str="SinPositionalEncoding",
                       norm_first:bool=True,
                       blotch_p:int=0.3,
                       drop_p:float=0.5,
                       *args, **kwargs):
        """
        n_tokens: int
            the number of tokens for the embedding layer
        d_model: int
            the number of dimensions for the latent vectors
        n_heads: int
            the number of attention heads
        h_mult: int
            a multiplier to determine the hidden dimensionality of the
            feed forward networks in the model.
        n_layers: int
            the number of transformer layers
        posenc_type: str
            the type of positional encoding. Argue the class name
            directly
        norm_first: bool
            if true, applies layer norm before the operations in the
            encoder layer (this seemed to be better in some paper I
            can't remember the name of)
        blotch_p: float
            the blotch probability. 0 means no blotching.
        drop_p: float
            the dropout probability. 0 means no dropout.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_mult = h_mult
        self.n_layers = n_layers
        self.blotch_p = blotch_p
        self.drop_p = drop_p
        self.posenc_type = posenc_type
        self.norm_first = norm_first

    def get_device(self):
        return next(self.parameters()).get_device()

class TransformerModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        self.embeddings = torch.nn.Embedding(self.n_tokens,self.d_model)
        self.pos_encoder = globals()[self.posenc_type](
            self.d_model, self.drop_p
        )
        d_hid = self.h_mult*self.d_model
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_model, self.n_heads, d_hid,
            self.drop_p, batch_first=True, norm_first=self.norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, self.n_layers
        )
        self.decoder = nn.Linear(self.d_model, self.n_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.embeddings.weight.data *= math.sqrt(self.d_model)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[batch_size, seq_len]``
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce

        Returns:
            output Tensor of shape ``[batch_size, seq_len, n_tokens]``
        """
        if not tforce: raise NotImplemented
        src = self.embeddings(src)
        src = self.pos_encoder(src)
        if mask is None:
            mask = generate_square_subsequent_mask(
                src.shape[1]
            ).to(self.get_device())
        elif is_causal:
            temp = generate_square_subsequent_mask(src.shape[1])
            mask = temp|mask
            mask = mask.to(self.get_device())
        output = self.transformer_encoder(
            src,
            mask=mask,
            src_key_padding_mask=pad_mask
        )
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generates an upper-triangular matrix of ``-inf``, with zeros on
    ``diag``.
    """
    #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        scale = (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * scale)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LossWrapper(torch.nn.Module):
    """
    This class wraps the model to keep the loss calculations distributed
    on all GPUs. Otherwise one gpu is overloaded with computational
    costs.
    """
    def __init__(self, model, tokenizer, hyps, *args, **kwargs):
        """
        loss_scale: float
            the loss is multiplied by this value on every iteration.
            useful as a way to normalize the learning rate when
            performing multiple gradient computations before each
            gradient step.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hyps = hyps
        self.loss_scale = 1./self.hyps.get("n_grad_loops",1)
        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.grad_clip = self.hyps.get("grad_clip", 10)

    def forward(self, data, ret_preds=False, seq_len=30,
                                             tforce=True,
                                             gen_targs=False,
                                             gen_ids=False,
                                             no_grad=False,
                                             temperature=1.,
                                             top_k=5):
        """
        Args:
            data: dict
                "input_ids": LongTensor (B,S1)
                    the token indices of the input sequence. The CMP
                    token should be appended to the end of each sentence.
                #"attention_mask": LongTensor (B,S1)
                #    attention mask for padding purposes. 0s mean padding.
                "output_ids": LongTensor (B,S2)
                    the token indices of the target sequence. An EOS
                    token should be appended to the end of each sentence
                #"output_attn_mask": LongTensor (B,S2)
                #    attention mask for padding purposes. 0s mean padding.
            seq_len: int
                the length of the output sequence.
            ret_preds: bool
                if true, will return the predictions
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            no_grad: bool
                if true, this function will not call .backward() on
                the loss. If false, this function will only call
                .backward if in training mode.
            top_k: int optional
                if argued, returns a calculation of the top_k accuracy
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,)
                "rmb_loss": torch tensor (1,)
                    only returned if `rmb_task` is true
                "acc": torch tensor (1,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
                "rmb_preds": torch tensor (B,S,P)
                    the rmb prediction logits. only returned if ret_preds
                    is true
                "top_k": torch tensor (1,)
                    the forward top n accuracy
                "rmb_top_k": torch tensor (1,)
                    the rmb top n accuracy.
        """
        ret_dict = dict()
        pad_mask = data["input_ids"]==self.tokenizer.pad_idx

        preds = self.model(
            data["input_ids"],
            pad_mask=pad_mask,
            is_causal=True,
            tforce=tforce
        )

        anti_pad_mask = ~pad_mask.reshape(-1)
        ps = preds.reshape(-1, preds.shape[-1])[anti_pad_mask]
        out_ids = data["output_ids"]
        labels = out_ids.reshape(-1)[anti_pad_mask]
        loss=self.loss_fxn(ps,labels)*self.loss_scale
        argmax = torch.argmax(ps, dim=-1)
        acc = (argmax==labels).float().mean()
        ret_dict["loss"] = loss
        ret_dict["acc"] = acc

        if self.training and not no_grad: loss.backward()

        if ret_preds: ret_dict["preds"] = preds

        return ret_dict

def blotch(idxs, sep_idx, blotch_p=0.4, indy_blotching=False):
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
        blotch_p: float
            the probability of blotching a valid, blotchable sequence.
        indy_blotching: bool
            one can make the mask such that the same blotching pattern
            is used for all tokens in a sequence, or blotching patterns
            can be unique to each token. If this value is true, the
            function will return a mask of shape (batch_size, seq_len,
            seq_len). Otherwise it will return a mask of shape
            (batch_size, seq_len)
    Returns:
        blotch_mask: torch BoolTensor (B,S,S) or (B,S)
            the shape will depend on the argument for indy_blotching.
    """
    seps = idxs==sep_idx
    sep_coords = torch.nonzero(seps)
    if indy_blotching:
        raise NotImplemented
        b,s = idxs.shape
        mask = torch.zeros(b,s,s,device=idxs.get_device()).bool()
        do_blotches = torch.rand(len(sep_coords), s)<blotch_p
        for i in range(len(do_blotches)-1):
            if sep_coords[i+1][0]==sep_coords[i][0]:
                row,col = sep_coords[i]
                _, stop_col = sep_coords[i+1]
                mask[row, col:stop_col] = do_blotches[row]
    else:
        mask = torch.zeros_like(idxs).bool()
        do_blotches = torch.rand(len(sep_coords))<blotch_p
        for i in range(len(do_blotches)-1):
            if do_blotches[i] and sep_coords[i+1][0]==sep_coords[i][0]:
                row,col = sep_coords[i]
                _,stop_col = sep_coords[i]
                mask[row, col:stop_col] = True
    return mask

def loss_and_acc(preds, labels, attn, loss_fxn, loss_scale=1,top_k=None):
    """
    preds: torch float tensor (B,S,L)
        prediction logits
    labels: torch long tensor (B,S)
        prediction ids
    attn: torch tensor (B,S)
        padding mask. 1s mean include these tokens, 0 means ignore them
    loss_fxn: function
        the loss function for the predictions
    loss_scale: float
        a scalar that scales the loss
    top_k: int optional
        if argued, returns a calculation of the top_k accuracy
    """
    ps = preds.reshape(-1,preds.shape[-1])
    device = ps.get_device()
    try:
        labels = labels.reshape(-1).to(device)
        idx = attn.bool().reshape(-1).to(device)
    except:
        device = "cpu"
        labels = labels.reshape(-1).to(device)
        idx = attn.bool().reshape(-1).to(device)
    argmax = torch.argmax(ps[idx], dim=-1)
    ret_dict = {
        "loss": loss_fxn(ps[idx],labels[idx])*loss_scale,
        "acc": (argmax==labels[idx]).float().mean(),
        "top_k": torch.zeros(1),
    }
    if top_k is not None:
        ret_dict["top_k"] = utils.top_k_acc(
            ps[idx], labels[idx], top_k, as_tensor=True
        )
    return ret_dict

