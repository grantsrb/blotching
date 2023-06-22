from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import get_blotch_mask, get_pos_ids
import ml_utils
import math

from transformers import (
    CONFIG_MAPPING,
    GPT2Config,
    AutoModelForCausalLM,
)

DEVICES = {
    -1: "cpu", **{i:i for i in range(10)}
}

class Model(torch.nn.Module):
    def __init__(self, n_tokens: int,
                       d_model:int=128,
                       n_heads:int=4,
                       h_mult:int=4,
                       n_layers:int=3,
                       posenc_type:str="SinPositionalEncoding",
                       norm_first:bool=True,
                       blotch_p:float=0,
                       blotch_p_max:float=None,
                       blotch_p_gran:int=None,
                       n_btokens:int=None,
                       bp_token_distr:str="uniform",
                       drop_p:float=0.5,
                       max_posencs:int=1000,
                       posenc_drop_p:float=None,
                       learn_posencs:bool=False,
                       pad_pos_skip:bool=False,
                       sep_idx:int=None,
                       actv_fxn:str="relu",
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
        blotch_p_max: float
            the highest blotch probability for the blotch tokens. 1
            means all blotching.
        blotch_p_gran: int
            the granularity of the blotch p the blotch token values will
            be divided by this value to determine the actual blotch p
        n_btokens: int or None
            the number of blotching tokens. If none or if `blotch_p_gran`
            is not None, will be decided by `blotch_p_gran`
        bp_token_distr: str { "uniform", "zipf_<n>", "poisson", "linear" }
            the distribution by which to sample the blotch_p tokens.
            Any arguments that include "zipf_" will use the following
            value as the exponent for a zipfian distribution.
            linear will sample proportionally to max_token-token. This
            is a linear function
        n_btokens: int
            the number of blotch tokens. This is effectively a
            granularity parameter for blotch values. If None, will
            default to blotch_p increments of 0.1 on the difference
            of bp_max and bp_min. For example, if bp_max-bp_min is
            0.4, then there will be 0.4/0.1 + 1 = 5 tokens
        drop_p: float
            the dropout probability. 0 means no dropout.
        max_posencs: int
            the number of possible embeddings. If
        posenc_drop_p: float optional
            the dropout probability for positional encodings. 0 means
            no dropout. defaults to drop_p if none
        learn_posencs: bool
            determines whether or not gradients are backpropagated into
            the positional encodings.
        pad_pos_skip: bool
            if true, will skip over masked tokens when applying positional
            encodings based on the pad mask.
        sep_idx: int
            the id of the sep token
        actv_fxn: str
            the transformer activation function
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_mult = h_mult
        self.n_layers = n_layers
        self.blotch_p = blotch_p
        self.bp_max  = blotch_p_max
        self.bp_gran = blotch_p_gran 
        self.n_btokens = n_btokens
        #self.bp_token_distr = self.get_bp_distr_fxn(bp_token_distr)
        if blotch_p_max:
            if self.bp_gran:
                # Add 1 to include both 0 and max
                self.n_btokens = max(int(self.bp_max*self.bp_gran)+1, 2)
            elif self.n_btokens:
                self.bp_gran = (self.n_btokens-1)/self.bp_max
            print("Num Blotch Tokens:", self.n_btokens)
            print(
                "Possible Bps:",
                torch.arange(self.n_btokens).float()/self.bp_gran
            )
        else:
            self.n_btokens = n_btokens if n_btokens else 0
        self.drop_p = drop_p
        self.posenc_type = posenc_type
        self.norm_first = norm_first
        self.max_posencs = max_posencs
        self.posenc_drop_p = posenc_drop_p
        if self.posenc_drop_p is None: self.posenc_drop_p = drop_p
        self.learn_posencs = learn_posencs
        self.pad_pos_skip = pad_pos_skip
        self.sep_idx = sep_idx
        self.actv_fxn = actv_fxn

    #def get_distr_fxn(self, distr):
    #    """
    #    Returns a sampling function to adjust the frequencies of
    #    particular tokens.

    #    Args:
    #        distr: str { "uniform", "zipf_<n>", "poisson", "linear" }
    #            the distribution by which to sample the blotch_p tokens.
    #            Any arguments that include "zipf_n" will use n as the
    #            exponent for a zipfian distribution. linear will sample
    #            proportionally to max_token-token
    #    """
    #    if "uniform"==distr: 
    #        return uniform_fxn(max_val=self.n_btokens)
    #    elif "zipf_" in distr:
    #        return zipf_fxn(
    #            order=float(distr.split("zipf_")[-1]),
    #            max_val=self.n_btokens
    #        )
    #    elif "poisson_" in distr:
    #        return poisson_fxn(float(distr.split("poisson_")[-1]))
    #    elif "linear"==distr:
    #        return linear_fxn(self.n_btokens)

    #def zipf_fxn(order, max_val):
    #    """
    #    Returns a sampling function with a sampling probability proportional
    #    to the argued order
    #    """

    def get_device(self):
        return next(self.parameters()).get_device()

    def get_blotch_ids(self, blotch_ps, eps=0.001):
        """
        This function takes a float or tensor of blotch probabilities
        and converts it into an id tensor corresponding to the blotch
        tokens. Remember that there is an offset of self.boffset, so
        the blotch id for a blotch_p of 0 is self.boffset

        Args:
            blotch_ps: float or FloatTensor (N,)
            eps: float
                a small value to avoid improper integer arithmetic.
                this value is added to the blotch_p x gran product
                to ensure the value surpasses the apprpriate threshold
        Returns:
            blotch_ids: LongTensor (N,)
        """
        if type(blotch_ps)!=type(torch.Tensor()):
            blotch_ps = torch.full((1,), blotch_ps)
        blotch_ids = torch.clamp(
            blotch_ps*self.bp_gran+eps, min=0, max=self.n_btokens-1
        ).long()
        blotch_ps = blotch_ids.float()/self.bp_gran
        return blotch_ids.long() + self.boffset, blotch_ps

    def sample_with_temperature(self, logits, temperature):
        if not temperature: return torch.argmax(logits, dim=-1)
        ps = torch.nn.functional.softmax( logits/temperature, dim=-1 )
        return torch.multinomial(ps, num_samples=1)

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
                      tforce:bool=True,
                      n_steps:int=10,
                      temperature=None,
                      incl_all_inpts=False,
                      blotch_p=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". Only applies if using freedom fwd. This is
                useful to save a concatenation during the data
                bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over tokens when applying positional
                encodings based on the pad mask.
            blotch_p: float
                the blotch probability. 0 means no blotching.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, n_tokens]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if tforce:
            if (self.blotch_p and self.training) or blotch_p:
                blotch_mask = get_blotch_mask(
                    src,
                    sep_idx=self.sep_idx,
                    blotch_p=self.blotch_p,
                )
                #pmask = pad_mask|blotch_mask
                #tok = kwargs["tokenizer"]
                #for i in range(5):
                #    print("before:", tok.decode(src[i][~pad_mask[i]]))
                #    print("after:", tok.decode(src[i][~pmask[i]]))
                #    print()
                #print()
                #print()

                pad_mask = pad_mask|blotch_mask

            ret_dict = self.tforce_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                *args, **kwargs,
            )
            if (self.blotch_p and self.training) or blotch_p:
                ret_dict["blotch_mask"] = blotch_mask
        else:

            ret_dict = self.freedom_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                incl_all_inpts=incl_all_inpts,
                temperature=temperature,
                *args, **kwargs,
            )
        return ret_dict


class HFModel(Model):
    """
    Uses a huggingface model base for the transformer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'TransformerModel'
        d_hid = self.h_mult*self.d_model
        config_kwargs = {
            "vocab_size": self.n_tokens+self.n_btokens,
            "hidden_size": self.d_model,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "n_ctx": self.max_posencs,
            "n_embd": self.d_model,
            "n_head": self.n_heads,
            "n_inner": d_hid,
            "activation_function": self.actv_fxn,
            "resid_pdrop": self.drop_p,
            "embd_pdrop": 0,
            "attn_pdrop": self.drop_p,
            "scale_attn_weights": True,
            "scale_attn_by_inverse_layer_idx": False,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
        }
        config = GPT2Config()
        config.update(config_kwargs)
        self.encoder = AutoModelForCausalLM.from_config(config)
        wpe = self.encoder.transformer.wpe
        for name, p in wpe.named_parameters():
            print("Turning off gradients for", name)
            p.requires_grad = self.learn_posencs
        pe = globals()[self.posenc_type](
            d_model=self.d_model,
            max_len=len(wpe.weight.data),
            learnable=self.learn_posencs,
            pad_pos_skip=self.pad_pos_skip
        )
        wpe.weight.data[:] = pe.pe.data[:len(wpe.weight.data)]

        self.embeddings = self.encoder.get_input_embeddings()
        self.decoder = nn.Linear( self.d_model, self.n_tokens )
        self.encoder.lm_head = self.decoder

        self.init_weights()

        self.register_buffer(
            "arange", torch.arange(self.max_posencs)
        )

    def tforce_fwd(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
                currently unused in this model type
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
        Returns:
            output Tensor of shape ``[bsize, seq_len, n_tokens]``
        """
        assert mask is None
        pmask = ~pad_mask.bool()
        pos_ids = get_pos_ids(
            pmask, arange=self.arange, pad_pos_skip=self.pad_pos_skip
        )
        output = self.encoder(
            input_ids=src,
            attention_mask=pmask,
            position_ids=pos_ids,
        )
        return {
            "logits": output.logits, "pred_ids": output.logits.argmax(-1)
        }

    def freedom_fwd(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      n_steps:int=10,
                      incl_all_inpts:bool=False,
                      temperature=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs.  This is useful to save a concatenation during
                the data bootstrapping phase.
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
        Returns:
            output Tensor of shape ``[bsize, seq_len+n_steps, n_tokens]``
        """
        assert mask is None
        B,S = src.shape
        n_loops = n_steps + 1

        ids = torch.nn.functional.pad(
            src, (0,n_loops), value=0
        )
        pad_mask = torch.nn.functional.pad(
            ~pad_mask.bool(), (0, n_loops), value=True
        )
        pos_ids = get_pos_ids(
            pad_mask, arange=self.arange, pad_pos_skip=self.pad_pos_skip
        )
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=DEVICES[src.get_device()]
        )
        logits[:,:S-1+incl_all_inpts].scatter_(
            dim=-1,
            index=src[:, 1-incl_all_inpts:S, None],
            src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
        )

        past_key_values = None
        inpt = ids[:,:S]
        pids = pos_ids[:,:S]
        mask = pad_mask[:,:S]
        for step in range(n_loops):
            ret = self.encoder(
                inpt,
                attention_mask=mask,
                position_ids=pids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = ret.past_key_values
            pred = ret.logits[:,-1]
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            ids[:,S+step] = argmaxs

            inpt = ids     [:,S+step:S+step+1]
            pids = pos_ids [:,S+step:S+step+1]
            mask = pad_mask[:,:S+step+1]
        return {
          "logits": logits, "pred_ids": ids[:,int(not incl_all_inpts):]
        }


class HFBlotchTokenModel(HFModel):
    """
    This model type includes a special token type indicating the
    blotch_p quantity. The token type is handled automatically by
    the forward function. Simply argue the appropriate blotch_p in the
    forward function. Generally speaking, the blotch_p will be rounded
    to the nearest 0.1, and you cannot exceed the blotch_p_max value
    set at the start.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Blotch'
        self.boffset = self.n_tokens

    def forward(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True,
                      n_steps:int=10,
                      temperature=None,
                      incl_all_inpts=False,
                      blotch_p=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Bool Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Bool Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". Only applies if using freedom fwd. This is
                useful to save a concatenation during the data
                bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over tokens when applying positional
                encodings based on the pad mask.
            blotch_p: float
                the amount of blotching to use. If None is argued, the
                blotching is sampled from bp_min to bp_max
                member variables.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, n_tokens]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        # Get appropriate blotch ids
        if blotch_p is None: blotch_p = torch.rand(len(src))*self.bp_max
        blotch_ids, blotch_p = self.get_blotch_ids(blotch_p)
        if len(blotch_ids)!=len(src):
            blotch_ids = torch.full((len(src),), blotch_ids.item())
            blotch_p = torch.full((len(src),), blotch_p.item())
        blotch_ids = blotch_ids.to(DEVICES[src.get_device()])

        ## TODO DELETME
        #ids = blotch_ids.cpu().numpy()-self.boffset
        #bp = blotch_p.cpu().numpy()
        #print("blotch_p", np.min(bp), np.max(bp))
        #print("bids", set(ids))
        #hist = {i:0 for i in set(ids)}
        #for i in ids: hist[i] += 1
        #s = sum(list(hist.values()))
        #keys = sorted(list(hist.keys()))
        #for k in keys:
        #    print(k, hist[k]/s)
        #print()

        if tforce:
            blotch_mask = get_blotch_mask(
                src,
                sep_idx=self.sep_idx,
                blotch_p=blotch_p,
            )

            pad_mask = pad_mask|blotch_mask.to(DEVICES[pad_mask.get_device()])

            pad_mask = torch.nn.functional.pad(
                pad_mask, (1, 0), value=False
            )
            src = torch.cat([blotch_ids[...,None], src], dim=-1)

            ret_dict = self.tforce_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                *args, **kwargs,
            )
            ret_dict["blotch_mask"] = blotch_mask
            # Remove the blotch token
            ret_dict["logits"] = ret_dict["logits"][:,1:]
            ret_dict["pred_ids"] = ret_dict["pred_ids"][:,1:]
        else:

            src = torch.cat([blotch_ids[:,None], src], dim=-1)
            pad_mask = torch.nn.functional.pad(
                pad_mask, (1, 0), value=False
            )

            ret_dict = self.freedom_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                incl_all_inpts=False,
                temperature=temperature,
                *args, **kwargs,
            )
            # Don't need to remove blotch token because freedom_fwd
            # always does it for us.
            if not incl_all_inpts:
                ret_dict["logits"] = ret_dict["logits"][:,1:]
                ret_dict["pred_ids"] = ret_dict["pred_ids"][:,1:]
        return ret_dict

class TransformerModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        self.embeddings = torch.nn.Embedding(
            self.n_tokens+self.n_btokens,self.d_model
        )
        self.pos_encoder = globals()[self.posenc_type](
            d_model=self.d_model,
            posenc_drop_p=self.posenc_drop_p,
            drop_p=self.drop_p,
            max_len=self.max_posencs,
            learnable=self.learn_posencs,
            pad_pos_skip=self.pad_pos_skip
        )
        d_hid = self.h_mult*self.d_model

        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_model,
            self.n_heads,
            d_hid,
            self.drop_p,
            batch_first=True,
            norm_first=self.norm_first
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.n_layers
        )
        self.decoder = nn.Linear(self.d_model, self.n_tokens)

        self.init_weights()

    def tforce_fwd(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
        Returns:
            output Tensor of shape ``[bsize, seq_len, n_tokens]``
        """
        embs = self.embeddings(src)
        if mask is None:
            mask = generate_square_subsequent_mask(
                embs.shape[1]
            ).to(self.get_device())
        elif is_causal:
            temp = generate_square_subsequent_mask(embs.shape[1])
            mask = temp|mask
            mask = mask.to(self.get_device())
        embs = self.pos_encoder( embs, mask=pad_mask )
        output = self.encoder(
            embs,
            mask=mask,
            src_key_padding_mask=pad_mask
        )
        logits = self.decoder(output)
        return { "logits": logits, "pred_ids": logits.argmax(-1) }

    def freedom_fwd(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      n_steps:int=10,
                      incl_all_inpts:bool=False,
                      pad_pos_skip:bool=False,
                      temperature=None):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". "predicted spaces" includes the shifted initial
                inputs.  This is useful to save a concatenation during
                the data bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over masked tokens when applying
                positional encodings based on the pad mask. True values
                in the mask will be skipped.
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
        Returns:
            output Tensor of shape ``[bsize, seq_len+n_steps, n_tokens]``
        """

        embs = self.embeddings(src)
        B,S,E = embs.shape
        n_loops = n_steps + 1

        pad_mask = torch.nn.functional.pad(
            pad_mask, (0, n_loops), value=False
        )
        ids = torch.nn.functional.pad(
            src, (0, n_loops), value=0
        )
        embs = torch.nn.functional.pad(
            embs, (0,0,0,n_loops), value=0
        )
        logits = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=DEVICES[embs.get_device()]
        )
        logits[:,:S-1+incl_all_inpts].scatter_(
            dim=-1,
            index=src[:, 1-incl_all_inpts:S, None],
            src=torch.ones_like(logits[:, :S-1+incl_all_inpts])
        )
        if mask is None:
            mask = generate_square_subsequent_mask(
                embs.shape[1]
            ).to(DEVICES[self.get_device()])
        elif is_causal:
            temp = generate_square_subsequent_mask(embs.shape[1])
            mask = temp|mask
            mask = mask.to(DEVICES[self.get_device()])

        for step in range(n_loops):
            temp = self.pos_encoder(
                embs[:,:S+step], mask=pad_mask[:,:S+step]
            )
            output = self.encoder(
                temp,
                mask=mask[:S+step, :S+step],
                src_key_padding_mask=pad_mask[:,:S+step]
            )
            pred = self.decoder(output[:,-1])
            logits[:,S-1+step+incl_all_inpts] = pred
            argmaxs = self.sample_with_temperature(
                pred, temperature
            ).squeeze()
            ids[:,S+step] = argmaxs
            embs[:,S+step] = self.embeddings(argmaxs)
        return {
            "logits": logits, "pred_ids": ids[:,int(not incl_all_inpts):]
        }

class BlotchTokenModel(TransformerModel):
    """
    This model type includes a special token type indicating the
    blotch_p quantity. The token type is handled automatically by
    the forward function. Simply argue the appropriate blotch_p in the
    forward function. Generally speaking, the blotch_p will be rounded
    to the nearest 0.1, and you cannot exceed the blotch_p_max value
    set at the start.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Blotch'
        self.boffset = self.n_tokens

    def forward(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True,
                      n_steps:int=10,
                      temperature=None,
                      incl_all_inpts=False,
                      blotch_p=None,
                      *args, **kwargs):
        """
        Arguments:
            src: Tensor, shape ``[bsize, seq_len]``
            mask: Bool Tensor, shape ``[seq_len, seq_len]``
            pad_mask: Bool Tensor, shape ``[bsize, seq_len]``
                true means padding
            is_causal: bool
                If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product
                attention.
            tforce: bool
                determines whether or not to teacherforce
            n_steps: int
                the number of prediction steps if not using teacher
                forcing
            temperature: float
                a parameter to adjust the entropy of the
                token sampling. high temperature means high entropy
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". Only applies if using freedom fwd. This is
                useful to save a concatenation during the data
                bootstrapping phase.
            pad_pos_skip: bool
                if true, will skip over tokens when applying positional
                encodings based on the pad mask.
            blotch_p: float
                the amount of blotching to use. If None is argued, the
                blotching is sampled from bp_min to bp_max
                member variables.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, boffset]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,boffset]``
        """
        # Get appropriate blotch ids
        if blotch_p is None: blotch_p = torch.rand(len(src))*self.bp_max
        blotch_ids, blotch_p = self.get_blotch_ids(blotch_p)
        if len(blotch_ids)!=len(src):
            blotch_ids = torch.full((len(src),), blotch_ids.item())
            blotch_p = torch.full((len(src),), blotch_p.item())
        blotch_ids = blotch_ids.to(DEVICES[src.get_device()])


        if tforce:
            blotch_mask = get_blotch_mask(
                src,
                sep_idx=self.sep_idx,
                blotch_p=blotch_p,
            )
            pad_mask = pad_mask|blotch_mask.to(DEVICES[pad_mask.get_device()])
            pad_mask = torch.nn.functional.pad(
                pad_mask, (1, 0), value=False
            )
            src = torch.cat([blotch_ids[...,None], src], dim=-1)

            ret_dict = self.tforce_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal
            )
            ret_dict["blotch_mask"] = blotch_mask
            # Remove the blotch token
            ret_dict["logits"] = ret_dict["logits"][:,1:]
            ret_dict["pred_ids"] = ret_dict["pred_ids"][:,1:]
        else:
            src = torch.cat([blotch_ids[...,None], src], dim=-1)
            pad_mask = torch.nn.functional.pad(
                pad_mask, (1, 0), value=False
            )
            ret_dict = self.freedom_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                incl_all_inpts=False,
                temperature=temperature,
            )
            # Don't need to remove blotch token because freedom_fwd
            # always does it for us.
            if not incl_all_inpts:
                ret_dict["logits"] = ret_dict["logits"][:,1:]
                ret_dict["pred_ids"] = ret_dict["pred_ids"][:,1:]
        return ret_dict


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag.
    """
    #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class IdentityPositionalEncoding(nn.Module):
    def __init__(self,
                 drop_p:float=0.1,
                 *args, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.dropout( x )
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000):
        super().__init__()
        self.posenc_dropout = nn.Dropout(p=posenc_drop_p)
        self.dropout = nn.Dropout(p=drop_p)
        self.arange = np.arange(max_len).astype("int")

    def rand_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        x = self.dropout( x + self.posenc_dropout(self.pe[idxs]) )
        return x

    def skip_rand_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.rand_forward(x)
        # pe: N, E
        n = np.random.randint(x.size(1), self.pe.shape[0]+1)
        idxs = torch.sort(torch.randperm(n)[:x.size(1)]).values.long()
        pe = self.posenc_dropout(self.pe[idxs])

        sums = (~mask).float().sum(-1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

    def vanil_forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = self.dropout( x + self.posenc_dropout(self.pe[:x.size(1)]) )
        return x

    def skip_vanil_forward(
            self,
            x: Tensor,
            mask: Tensor,
            *args,
            **kwargs
        ) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            mask: Tensor, shape ``[batch_size, seq_len]``
                pad mask. true values represent padding/blotching
        """
        if mask is None: return self.vanil_forward(x)
        pe = self.posenc_dropout(self.pe[:x.size(1)])

        sums = torch.sum((~mask).float(), -1)
        idxs = torch.cat([torch.arange(s) for s in sums], axis=0).long()
        fx = torch.zeros_like(x)
        fx[~mask] += pe[idxs]
        fx = x + fx

        return self.dropout( fx )

class RandPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        pe = 0.1*math.sqrt(max_len/d_model)*torch.randn(max_len,d_model)
        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward

class SinPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 d_model:int,
                 posenc_drop_p:float=0,
                 drop_p:float=0.1,
                 max_len:int=1000,
                 learnable:bool=False,
                 pad_pos_skip:bool=False):
        super().__init__(posenc_drop_p, drop_p, max_len=max_len)
        self.pad_pos_skip = pad_pos_skip

        position = torch.arange(max_len).unsqueeze(1)
        scale = (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * scale)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if learnable: self.pe = torch.nn.Parameter(pe)
        else: self.register_buffer('pe', pe)

        if pad_pos_skip:
            self.forward = self.skip_vanil_forward
        else:
            self.forward = self.vanil_forward


class RandSinPositionalEncoding(SinPositionalEncoding):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.pad_pos_skip:
            self.forward = self.skip_rand_forward
        else:
            self.forward = self.rand_forward


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
        self.loss_fxn = torch.nn.functional.cross_entropy
        self.grad_clip = self.hyps.get("grad_clip", 10)

    def forward(self, data, ret_preds=False, tforce=True,
                                             no_grad=False,
                                             prob_len=None,
                                             incl_intl_prob=False,
                                             temperature=None,
                                             incl_all_inpts=False,
                                             top_k=5,
                                             blotch_p=None,
                                             reduce_metrics=True,
                                             *args, **kwargs):
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
            ret_preds: bool
                if true, will return the predictions
            tforce: bool
                determines whether model should use teacher forcing for
                predictions or not.
            incl_intl_prob: bool
                if true, will include the initial problem in the loss.
                if false, will exclude initial problem from the loss.
            temperature: float
                a temperature parameter for softmax sampling. Set to
                low number for high confidence sampling, high value
                for low confidence sampling
            incl_all_inpts: bool
                if true, will include all input tokens in the output
                prediction tensor. otherwise only includes "predicted
                spaces". Only applies if using freedom fwd. This is
                useful to save a concatenation during the data
                bootstrapping phase.
            prob_len: int or None
                the index at which the problem is separated from
                the solution. If none, it is found via torch indexing.
            no_grad: bool
                if true, this function will not call .backward() on
                the loss. If false, this function will only call
                .backward if in training mode.
            top_k: int optional
                if argued, returns a calculation of the top_k accuracy
            blotch_p: float or float tensor (B,)
            reduce_metrics: bool
                if true, loss and acc will be averaged over all samples.
                if false, loss and acc will be returned as tensors for
                each token prediction
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,) or (B,)
                "acc": torch tensor (1,) or (B,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S)
                    the prediction ids
        """
        ret_dict = dict()
        pad_idx = self.tokenizer.pad_idx
        eos_idx = self.tokenizer.eos_idx
        if "inpt_pad_mask" not in data:
            inpt_pad_mask = (data["input_ids"]==pad_idx)
            inpt_pad_mask = inpt_pad_mask|(data["input_ids"]==eos_idx)
        else: inpt_pad_mask = data["inpt_pad_mask"].clone()
        if "out_pad_mask" not in data:
            out_pad_mask  = data["output_ids"]==pad_idx
        else: out_pad_mask = data["out_pad_mask"].clone()

        # Need to be careful with intermediate padding
        if tforce:
            ret_dict = self.model(
                data["input_ids"],
                pad_mask=inpt_pad_mask,
                is_causal=True,
                tforce=tforce,
                blotch_p=blotch_p,
                tokenizer=self.tokenizer
            )
            logits = ret_dict["logits"]
            if "blotch_mask" in ret_dict:
                blotch_mask = ret_dict["blotch_mask"]
                inpt_pad_mask = inpt_pad_mask|blotch_mask
                out_pad_mask[:,:-1]=out_pad_mask[:,:-1]|blotch_mask[:,1:]
        else:
            if prob_len is None:
                s = self.tokenizer.sep_idx
                prob_len = torch.argmax(data["input_ids"][0]==s,dim=-1)
            # +1 to include intial equals sign in seed sequence
            plen = prob_len + 1
            tot_len = data["output_ids"].shape[-1]
            ret_dict = self.model(
                data["input_ids"][...,:plen],
                pad_mask=inpt_pad_mask[..., :plen],
                is_causal=True,
                tforce=tforce,
                n_steps=tot_len-plen,
                incl_all_inpts=incl_all_inpts,
                blotch_p=blotch_p,
                tokenizer=self.tokenizer
            )
            logits = ret_dict["logits"]

        if not incl_intl_prob:
            if prob_len is None:
                s = self.tokenizer.sep_idx
                prob_len = torch.argmax(data["input_ids"][0]==s,dim=-1)
            inpt_pad_mask[...,:prob_len] = True
            out_pad_mask [...,:prob_len] = True

        #print()
        #print("Full inpt:",
        #  self.tokenizer.decode(data["input_ids"][0]))
        #print("Full Outpt:",
        #  self.tokenizer.decode(data["output_ids"][0]))
        #print("dropped inpt:",
        #  self.tokenizer.decode(data["input_ids"][0][inpt_pad_mask[0]]))
        #print("dropped out:",
        #  self.tokenizer.decode(data["output_ids"][0][out_pad_mask[0]]))
        #print("post inpt:",
        #  self.tokenizer.decode(data["input_ids"][0][~inpt_pad_mask[0]]))
        #print("post out:",
        #  self.tokenizer.decode(data["output_ids"][0][~out_pad_mask[0]]))
        #print()
        #print()
        #print()
        #print()

        out_ids = data["output_ids"]
        inpt_mask = ~inpt_pad_mask.reshape(-1)
        out_mask =  ~out_pad_mask.reshape(-1)
        ps = logits[:,int(incl_all_inpts):].reshape(
            -1, logits.shape[-1]
        )[inpt_mask]
        labels = out_ids.reshape(-1)[out_mask]
        reduction = "mean" if reduce_metrics else "none"
        loss=self.loss_fxn(ps,labels,reduction=reduction)*self.loss_scale
        if self.training and not no_grad: loss.backward()
        elif not reduce_metrics:
            temp = torch.zeros_like(out_ids).float()
            temp[out_mask.reshape(out_ids.shape)] = loss
            loss = temp
        ret_dict["loss"] = loss

        argmax = torch.argmax(ps, dim=-1)
        acc = (argmax==labels).float()
        if reduce_metrics: acc = acc.mean()
        ret_dict["acc"] = acc

        out_ids = data["output_ids"]
        out_ends = torch.argmax( (out_ids==eos_idx).long(), dim=-1 )
        # Case where soln len exceeds seq_len. Ideally this doesn't
        # happen
        out_ends[out_ends==0] = out_ends.shape[-1]-1 
        pred_ids = logits[:,int(incl_all_inpts):].argmax(-1)
        pred_ids[:,-1] = eos_idx
        pred_ends = torch.argmax( (pred_ids==eos_idx).long(), dim=-1 )
        diffs = (out_ends-pred_ends).float()
        ret_dict["len_diff"] = diffs.mean()
        ret_dict["len_percent"] = (diffs/out_ends).mean()*100

        # Replace predictions with ground truth if not training on
        # initial prob
        if not incl_intl_prob and tforce:
            ids = data["output_ids"][...,:prob_len]
            ret_dict["pred_ids"][...,:prob_len] = ids
        return ret_dict

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
    device = DEVICES[ps.get_device()]
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
        ret_dict["top_k"] = ml_utils.top_k_acc(
            ps[idx], labels[idx], top_k, as_tensor=True
        )
    return ret_dict

if __name__=="__main__":
    #def blotch(idxs, sep_idx, blotch_p=0.4, indy_blotching=False):
    sep_idx = 0
    blotch_p = 0.25
    n_samples = 5000
    slen = 9
    allow_contig = True

    idxs = torch.randint(2,9, size=(4,slen))
    idxs[:2,:-1:2] = sep_idx
    idxs[2:,1:-1:2] = sep_idx
    print("Idxs:", idxs)

    blotches = torch.ones(len(idxs))*torch.FloatTensor([
        0.1, 0.3, 0.6, 0.9
    ])
    n_seps = torch.max(torch.sum(idxs==sep_idx, dim=1))
    for i in range(n_seps):
        bmask = get_blotch_mask(
            idxs,
            sep_idx=sep_idx,
            #blotch_p=blotches,
            allow_contig=allow_contig,
            step_idx = i
        )
        print("blotch:", idxs[bmask])
    #for row in range(idxs.shape[0]):
    #    print()
    #    print("Unblotched:", idxs[row])
    #    print("Mask:", bmask[row])
    #    print("Blotched:", idxs[row][~bmask[row]])
    #print("bmask p:", bmask.float().sum()/2/(idxs==sep_idx).float().sum())

    #idxs = torch.randint(2,9, size=(n_samples,slen))
    #idxs[:2,:-1:2] = sep_idx
    #idxs[2:,1:-1:2] = sep_idx

    #blotches = torch.ones(len(idxs))*blotch_p
    #bmask = get_blotch_mask(
    #    idxs,
    #    sep_idx=sep_idx,
    #    blotch_p=blotches,
    #    allow_contig=allow_contig
    #)
    #print("bmask p:", bmask.float().sum()/2/(idxs==sep_idx).float().sum())




