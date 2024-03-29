from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import utils
from utils import get_blotch_mask, get_pos_ids, get_tok_mask, arglast
import ml_utils
import math

from transformers import (
    CONFIG_MAPPING,
    GPT2Config,
    AutoModelForCausalLM,
    OpenAIGPTConfig,
    GPTJConfig,
    LlamaConfig,
    TransfoXLConfig,
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
                       blotch_spacing:str="random",
                       blotch_p:float=0.3,
                       blotch_p_min:float=None,
                       blotch_p_max:float=None,
                       tok_drop_p:float=0,
                       n_btokens:int=None,
                       drop_p:float=0.5,
                       max_posencs:int=1000,
                       posenc_drop_p:float=None,
                       learn_posencs:bool=False,
                       pad_pos_skip:bool=False,
                       sep_idx:int=None,
                       actv_fxn:str="gelu",
                       scale_attn_weights:bool=True,
                       scale_by_inv_layer:bool=True,
                       reorder_and_upcast:bool=False,
                       hf_model_type:str="gpt2",
                       excise_tokens:bool=False,
                       pretrained:bool=False,
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
        blotch_spacing: str
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
        blotch_p: float
            the blotch probability. 0 means no blotching.
        blotch_p_min: float
            the lowest blotch probability for the blotch tokens. 0
            means no blotching.
        blotch_p_max: float
            the highest blotch probability for the blotch tokens. 1
            means all blotching.
        tok_drop_p: float
            the probability of randomly dropping a token. 0 means no
            token dropping.
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
        hf_model_type: str
            the huggingface transformer base. only applies if using
            HFModel types. Specifies the hf model base type.
        excise_tokens: bool
            if true, instead of rearranging the positional encoding
            ids, the blotched tokens will actually be removed from
            the input. This makes it easier to use relative encodings.
        scale_attn_weights: bool
            scale attention weights by dividing by sqrt(hidden_size)
        scale_by_inv_layer: bool
            scale attention weights by inverse layer index. see
            huggingface docs for details
        reorder_and_upcast: bool
            reorder and upcast attention. see huggingface docs for details
        pretrained: bool
            if true, will ignore model specs and use a pretrained
            huggingface model. only applies if using HF model types.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_heads = n_heads
        self.h_mult = h_mult
        self.n_layers = n_layers
        self.bp_spacing = blotch_spacing
        self.blotch_p = blotch_p
        self.tok_drop_p = tok_drop_p
        self.bp_min = blotch_p_min
        if blotch_p_min is None: self.bp_min = blotch_p
        self.bp_max = blotch_p_max
        if blotch_p_max is None: self.bp_max = blotch_p
        self.bp_diff = self.bp_max - self.bp_min
        self.n_btokens = n_btokens if self.bp_diff>0 else None
        if self.n_btokens is None:
            self.bp_gran = 11 # the granularity of the blotch p
            # the blotch token values will be divided by this value
            # to determine the actual blotch p
            self.n_btokens = max(int(self.bp_diff*self.bp_gran), 1)
        else:
            assert self.bp_diff>0
            self.bp_gran = self.n_btokens/self.bp_diff
        print("Num Blotch Tokens:", self.n_btokens)
        if self.n_btokens:
            print(
                "Possible Blotch Ps:",
                torch.arange(self.n_btokens)/self.bp_gran
            )
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
        self.hf_model_type = hf_model_type
        self.scale_attn_weights = scale_attn_weights
        self.scale_by_inv_layer = scale_by_inv_layer
        self.reorder_and_upcast = reorder_and_upcast
        self.excise_tokens = excise_tokens
        self.pretrained = pretrained

    def get_device(self):
        return DEVICES[next(self.parameters()).get_device()]

    def get_blotch_ids(self, blotch_ps):
        """
        This function takes a float or tensor of blotch probabilities
        and converts it into an id tensor corresponding to the blotch
        tokens. Remember that there is an offset of self.n_tokens, so
        the blotch id for a blotch_p of 0 is self.n_tokens

        Args:
            blotch_ps: float or FloatTensor (N,)
        Returns:
            blotch_ids: LongTensor (N,)
        """
        if type(blotch_ps)!=type(torch.Tensor()):
            blotch_ps = torch.full((1,), blotch_ps)
        if self.bp_diff == 0: blotch_range = torch.zeros_like(blotch_ps)
        else: blotch_range = (blotch_ps+0.01-self.bp_min)/self.bp_diff
        blotch_ids = (blotch_range*self.n_btokens).long()+self.n_tokens
        return blotch_ids

    def sample_with_temperature(self, logits, temperature):
        """
        Args:
            logits: torch float tensor (..., L)
            temperature: float or None
                a value to increase the sampling entropy. ignored if
                0 or None
        """
        if not temperature: return torch.argmax(logits, dim=-1)
        ps = torch.nn.functional.softmax( logits/temperature, dim=-1 )
        return torch.multinomial(ps, num_samples=1)[...,0]

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
                      tok_drop_p=None,
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
            tok_drop_p: float
                the probability of randomly dropping a token. 0 means no
                token dropping.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, n_tokens]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if tforce:
            if (self.blotch_p and self.training) or blotch_p:
                if not blotch_p: blotch_p = self.blotch_p
                blotch_mask = get_blotch_mask(
                    src,
                    sep_idx=self.sep_idx,
                    blotch_p=blotch_p,
                    spacing=self.bp_spacing,
                )
                pad_mask = pad_mask|blotch_mask
            if (self.tok_drop_p and self.training) or tok_drop_p:
                if not tok_drop_p: tok_drop_p = self.tok_drop_p
                tok_mask = get_tok_mask(
                    src=src, tok_p=tok_drop_p, sep_idx=self.sep_idx
                )
                pad_mask = pad_mask|tok_mask

            ret_dict = self.tforce_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal
            )
            if blotch_p or (self.blotch_p and self.training):
                ret_dict["blotch_mask"] = blotch_mask
            if tok_drop_p or (self.tok_drop_p and self.training):
                ret_dict["tok_mask"] = tok_mask
        else:
            ret_dict = self.freedom_fwd(
                src=src,
                mask=mask,
                pad_mask=pad_mask,
                is_causal=is_causal,
                n_steps=n_steps,
                incl_all_inpts=incl_all_inpts,
                temperature=temperature,
            )
        return ret_dict


class TransformerModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'
        self.embeddings = torch.nn.Embedding(self.n_tokens,self.d_model)
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
        self.transformer_encoder = nn.TransformerEncoder(
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
        output = self.transformer_encoder(
            embs,
            mask=mask,
            src_key_padding_mask=pad_mask
        )
        return {"preds": self.decoder(output)}

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
        embs = torch.nn.functional.pad(
            embs, (0,0,0,n_loops), value=0
        )
        preds = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=DEVICES[embs.get_device()]
        )
        preds[:,:S-1+incl_all_inpts].scatter_(
            dim=-1,
            index=src[:, 1-incl_all_inpts:S, None],
            src=torch.ones_like(preds[:, :S-1+incl_all_inpts])
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
            output = self.transformer_encoder(
                temp,
                mask=mask[:S+step, :S+step],
                src_key_padding_mask=pad_mask[:,:S+step]
            )
            pred = self.decoder(output[:,-1])
            preds[:,S-1+step+incl_all_inpts] = pred
            if step < n_steps:
                argmaxs = self.sample_with_temperature(
                    pred, temperature
                ).squeeze()
                embs[:,S+step] = self.embeddings(argmaxs)
        return {"preds": preds}

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
        n_embs = self.n_tokens+self.n_btokens
        self.embeddings = torch.nn.Embedding(n_embs,self.d_model)
        self.init_weights()

    def forward(self, src:torch.Tensor,
                      mask:torch.Tensor=None,
                      pad_mask:torch.Tensor=None,
                      is_causal:bool=None,
                      tforce:bool=True,
                      n_steps:int=10,
                      temperature=None,
                      incl_all_inpts=False,
                      blotch_p=None,
                      tok_drop_p=None,
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
            tok_drop_p: float
                the probability of randomly dropping a token. 0 means no
                token dropping.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, n_tokens]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if blotch_p is not None:
            blotch_ids = self.get_blotch_ids(blotch_p)
            if len(blotch_ids) == 1:
                # Just creates a tensor of len(src) of all blotch_id values
                blotch_ids = torch.full((len(src),), blotch_ids.item())
            blotch_p = (blotch_ids-self.n_tokens).float()/self.bp_gran
        else:
            blotch_range = torch.rand(len(src))
            blotch_ids=(blotch_range*self.n_btokens).long()
            blotch_p = blotch_ids.float()/self.bp_gran
            #print()
            ##blotch_p = blotch_range*self.bp_diff+self.bp_min
            #print("No argued bp")
            #print("bids:", blotch_ids[:10])
            #print("bps:", blotch_p[:10])
            #print("bids unique:", torch.unique(blotch_ids))
            #print("bps unique:", torch.unique(blotch_p))
            #print("bid distr:")
            #hist = {i: (blotch_ids==i).float().mean() for i in range(11)}
            #for k in hist:
            #    print(k, "-", hist[k])
            blotch_ids += self.n_tokens
        blotch_ids = blotch_ids.to(DEVICES[src.get_device()])

        if tforce:
            blotch_mask = get_blotch_mask(
                src,
                sep_idx=self.sep_idx,
                blotch_p=blotch_p,
                spacing=self.bp_spacing,
            )
            pad_mask = pad_mask|blotch_mask.to(DEVICES[pad_mask.get_device()])

            if (self.tok_drop_p and self.training) or tok_drop_p:
                if not tok_drop_p: tok_drop_p = self.tok_drop_p
                tok_mask = get_tok_mask(
                    src=src, tok_p=tok_drop_p, sep_idx=self.sep_idx
                )
                pad_mask = pad_mask|tok_mask


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
            ret_dict["preds"] = ret_dict["preds"][:,1:]
            if (self.tok_drop_p and self.training) or tok_drop_p:
                ret_dict["tok_mask"] = tok_mask
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
                ret_dict["preds"] = ret_dict["preds"][:,1:]
        return ret_dict

class HFModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = 'Transformer'

        if self.pretrained:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                self.hf_model_type
            )
            print("Properties:")
            for name in dir(self.encoder):
                print(name)
            embeddings = self.encoder.get_input_embeddings()
            self.d_model = embeddings.weight.shape[-1]
            print(self.encoder)
            self.embeddings = torch.nn.Embedding(
                self.n_tokens,self.d_model
            )
        else:
            config = self.get_config()
            self.encoder = AutoModelForCausalLM.from_config(config)
            if hasattr(self.encoder, "transformer"):
                if hasattr(self.encoder.transformer, "wpe"):
                    wpe = self.encoder.transformer.wpe
                    for name, p in wpe.named_parameters():
                        print("Turning off gradients for", name)
                        p.requires_grad = self.learn_posencs
            self.embeddings = self.encoder.get_input_embeddings()

        self.decoder = nn.Linear( self.d_model, self.n_tokens )
        self.encoder.lm_head = self.decoder

        self.init_weights()

        self.register_buffer(
            "arange", torch.arange(self.max_posencs)
        )

    def get_config(self):
        """
        Finds the appropirate configuration when using Huggingface
        models.
        """
        d_hid = self.h_mult*self.d_model
        config_kwargs = {
            "vocab_size": self.n_tokens+self.n_btokens,
            "hidden_size": self.d_model,
            "intermediate_size": d_hid,
            "num_hidden_layers": self.n_layers,
            "num_attention_heads": self.n_heads,
            "num_key_value_heads": self.n_heads,
            "hidden_act": self.actv_fxn,
            "n_positions": self.max_posencs,
            "rotary_dim": self.d_model//self.n_heads,
            "rope_theta": self.d_model//self.n_heads,
            "n_ctx": self.max_posencs,
            "n_embd": self.d_model,
            "n_head": self.n_heads,
            "n_inner": d_hid,
            "activation_function": self.actv_fxn,
            "resid_pdrop": self.drop_p,
            "embd_pdrop":  0,
            "attn_pdrop":  self.drop_p,
            "scale_attn_weights": self.scale_attn_weights,
            "scale_attn_by_inverse_layer_idx": self.scale_by_inv_layer,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "reorder_and_upcast_attn": self.reorder_and_upcast,
            "add_cross_attention": False,
        }
        if self.hf_model_type=="gpt2":
            config = GPT2Config()
        elif self.hf_model_type == "gptj":
            config = GPTJConfig()
        elif self.hf_model_type == "llama":
            config = LlamaConfig()
        elif self.hf_model_type == "transxl":
            config = TransfoXLConfig()
        config.update(config_kwargs)
        return config

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
        pmask = ~(pad_mask.bool())
        # Actually removes the tokens at the pad_mask locations
        if self.excise_tokens:
            old_mask = pad_mask
            pad_id = src[0][pad_mask[0]].reshape(-1)[0].item()
            src,new_mask = utils.excise_tokens(
                src, ~pad_mask.bool(), pad_id=pad_id
            )
            pad_mask = src==pad_id
            pos_ids = None
        # Changes the pos_ids to make it seem like the tokens at the
        # pad_mask locations have been removed
        else:
            pos_ids = get_pos_ids(
                pmask, arange=self.arange, pad_pos_skip=self.pad_pos_skip
            )
        output = self.encoder(
            src,
            attention_mask=pmask,
            position_ids=pos_ids,
        )
        # Need to put the tokens back in the correct place for integration
        # with the loss functions
        if self.excise_tokens:
            output.logits = utils.reverse_excision(
                output.logits, new_mask, old_mask
            )
        return { "preds": output.logits }

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
                inputs. This is useful to save a concatenation during
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
        B,S = src.shape
        n_loops = n_steps + 1

        pad_mask = torch.nn.functional.pad(
            ~(pad_mask.bool()), (0, n_loops), value=True
        )
        ids = torch.zeros_like(pad_mask).long()
        ids[:,:S] = src
        pos_ids = get_pos_ids(
            pad_mask, arange=self.arange, pad_pos_skip=self.pad_pos_skip
        )
        preds = torch.zeros(
            (B,S+n_steps+incl_all_inpts,self.n_tokens),
            device=DEVICES[pad_mask.get_device()]
        )
        preds[:,:S-1+incl_all_inpts].scatter_(
            dim=-1,
            index=src[:, 1-incl_all_inpts:S, None],
            src=torch.ones_like(preds[:, :S-1+incl_all_inpts])
        )

        past_key_values = None
        inpt = ids[:,:S]
        pids = pos_ids[:,:S]
        for step in range(n_loops):
            output = self.encoder(
                inpt,
                attention_mask=pad_mask[:,:S+step],
                position_ids=pids,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output.past_key_values
            pred = output.logits[:,-1]
            preds[:,S-1+step+incl_all_inpts] = pred
            if step < n_steps:
                argmaxs = self.sample_with_temperature(
                    pred, temperature
                ).squeeze()
                ids[:,S+step] = argmaxs
                inpt = ids[:,S+step:S+step+1]
                pids = pos_ids[:,S+step:S+step+1]
        return { "preds": preds }


class HFBlotchModel(HFModel):
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
                      tok_drop_p=None,
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
            tok_drop_p: float
                the probability of randomly dropping a token. 0 means no
                token dropping.
        Returns:
            if tforce:
              output Tensor of shape ``[bsize, seq_len, n_tokens]``
            else:
              output Tensor of shape ``[bsize,seq_len+n_steps,n_tokens]``
        """
        if blotch_p is not None:
            blotch_ids = self.get_blotch_ids(blotch_p)
            if len(blotch_ids) == 1:
                # Just creates a tensor of len(src) of all blotch_id values
                blotch_ids = torch.full((len(src),), blotch_ids.item())
            blotch_p = (blotch_ids-self.n_tokens).float()/self.bp_gran
        else:
            blotch_range = torch.rand(len(src))
            blotch_ids=(blotch_range*self.n_btokens).long()
            blotch_p = blotch_ids.float()/self.bp_gran
            blotch_ids += self.n_tokens
        blotch_ids = blotch_ids.to(DEVICES[src.get_device()])

        if tforce:
            blotch_mask = get_blotch_mask(
                src,
                sep_idx=self.sep_idx,
                blotch_p=blotch_p,
                spacing=self.bp_spacing,
            )
            pad_mask = pad_mask|blotch_mask.to(DEVICES[pad_mask.get_device()])

            if (self.tok_drop_p and self.training) or tok_drop_p:
                if not tok_drop_p: tok_drop_p = self.tok_drop_p
                tok_mask = get_tok_mask(
                    src=src, tok_p=tok_drop_p, sep_idx=self.sep_idx
                )
                pad_mask = pad_mask|tok_mask

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
            ret_dict["preds"] = ret_dict["preds"][:,1:]
            if (self.tok_drop_p and self.training) or tok_drop_p:
                ret_dict["tok_mask"] = tok_mask
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
                ret_dict["preds"] = ret_dict["preds"][:,1:]
        return ret_dict


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """
    Generates an upper-triangular matrix of True, with Falses on
    diag.
    """
    #return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

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
        self.label_smoothing = hyps.get("label_smoothing", 0)
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
                                             tok_drop_p=None,
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
            tok_drop_p: float
                the probability of randomly dropping a token. 0 means no
                token dropping.
        Returns:
            ret_dict: dict (keys: str, vals: torch tensor)
                "loss": torch tensor (1,) or (B,)
                "acc": torch tensor (1,) or (B,)
                    the raw accuracy for the non-rmb task
                "preds": torch tensor (B,S,P)
                    the prediction logits. only returned if ret_preds is
                    true
        """
        ret_dict = dict()
        pad_idx = self.tokenizer.pad_idx
        eos_idx = self.tokenizer.eos_idx
        if "inpt_pad_mask" not in data:
            inpt_pad_mask = (data["input_ids"]==pad_idx)
            inpt_pad_mask = inpt_pad_mask|(data["input_ids"]==eos_idx)
        else: inpt_pad_mask = data["inpt_pad_mask"].clone()
        if "out_pad_mask" not in data:
            out_pad_mask = data["output_ids"]==pad_idx
        else: out_pad_mask = data["out_pad_mask"].clone()

        # Need to be careful with intermediate padding
        if tforce:
            ret_dict = self.model(
                data["input_ids"],
                pad_mask=inpt_pad_mask,
                is_causal=True,
                tforce=tforce,
                blotch_p=blotch_p,
                tok_drop_p=tok_drop_p,
                tokenizer=self.tokenizer,
                temperature=temperature,
            )
            preds = ret_dict["preds"]
            if "blotch_mask" in ret_dict:
                blotch_mask = ret_dict["blotch_mask"]
                inpt_pad_mask = inpt_pad_mask|blotch_mask
                out_pad_mask[:,:-1]=out_pad_mask[:,:-1]|blotch_mask[:,1:]
            if "tok_mask" in ret_dict:
                tok_mask = ret_dict["tok_mask"]
                tok_mask[data["input_ids"]==eos_idx] = False
                inpt_pad_mask = inpt_pad_mask|tok_mask
                out_pad_mask[:,:-1]=out_pad_mask[:,:-1]|tok_mask[:,1:]
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
                tok_drop_p=tok_drop_p,
                tokenizer=self.tokenizer,
                temperature=temperature,
            )
            preds = ret_dict["preds"]
            #print(
            #    "input:",
            #    data["input_ids"][0,:plen][~inpt_pad_mask[0,:plen]]
            #)
            #out_ids = data["output_ids"]
            #print(
            #    "output:",
            #    out_ids[0,prob_len:][~out_pad_mask[0,prob_len:]]
            #)

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

        out_ids = data["output_ids"]
        inpt_mask = ~inpt_pad_mask.reshape(-1)
        out_mask =  ~out_pad_mask.reshape(-1)
        ps = preds[:,int(incl_all_inpts):].reshape(
            -1, preds.shape[-1]
        )[inpt_mask]
        labels = out_ids.reshape(-1)[out_mask]
        reduction = "mean" if reduce_metrics else "none"
        try:
            loss = self.loss_scale*self.loss_fxn(
                ps,labels,
                reduction=reduction,
                label_smoothing=self.label_smoothing
            )
        except:
            for i in range(len(data["input_ids"])):
                print()
                print("Full inpt:",
                  self.tokenizer.decode(data["input_ids"][i]))
                print("Full Outpt:",
                  self.tokenizer.decode(data["output_ids"][i]))
                print("dropped inpt:",
                  self.tokenizer.decode(
                    data["input_ids"][i][inpt_pad_mask[i]]))
                print("dropped out:",
                  self.tokenizer.decode(
                    data["output_ids"][i][out_pad_mask[i]]))
                print("post inpt:",
                  self.tokenizer.decode(
                    data["input_ids"][i][~inpt_pad_mask[i]]))
                print("post out:",
                  self.tokenizer.decode(
                    data["output_ids"][i][~out_pad_mask[i]]))

            idx = inpt_pad_mask.float().sum(-1)!=out_pad_mask.float().sum(-1)
            print()
            print()
            print()
            print()
            for i in range(idx.long().sum(-1)):
                print("Full inpt:",
                  self.tokenizer.decode(data["input_ids"][idx][i]))
                print("Full Outpt:",
                  self.tokenizer.decode(data["output_ids"][idx][i]))
                print("dropped inpt:",
                  self.tokenizer.decode(
                    data["input_ids"][idx][i][inpt_pad_mask[idx][i]]))
                print("dropped out:",
                  self.tokenizer.decode(
                    data["output_ids"][idx][i][out_pad_mask[idx][i]]))
                print("post inpt:",
                  self.tokenizer.decode(
                    data["input_ids"][idx][i][~inpt_pad_mask[idx][i]]))
                print("post out:",
                  self.tokenizer.decode(
                    data["output_ids"][idx][i][~out_pad_mask[idx][i]]))
            assert False
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
        sep_id = self.tokenizer.sep_idx
        out_lasts = arglast( (out_ids==sep_id).long(), dim=-1 )

        pred_ids = preds[:,int(incl_all_inpts):].argmax(-1)
        pred_lasts = arglast((pred_ids==sep_id).long(), dim=-1)
        diffs = (out_lasts-pred_lasts).float()
        ret_dict["len_diff"] = diffs.mean()
        ret_dict["len_percent"] = (diffs/out_lasts).mean()*100

        if ret_preds:
            ret_dict["preds"] = preds.argmax(-1)
            # Replace predictions with ground truth if not training on
            # initial prob
            if not incl_intl_prob and tforce:
                ids = data["output_ids"][...,:prob_len]
                ret_dict["preds"][...,:prob_len] = ids
        return ret_dict

def loss_and_acc(preds,
        labels,
        attn,
        loss_fxn,
        loss_scale=1,
        top_k=None,
        label_smoothing=0,
    ):
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
        "loss": loss_fxn(
                    ps[idx],labels[idx],
                    label_smoothing=label_smoothing
                )*loss_scale,
        "acc": (argmax==labels[idx]).float().mean(),
        "top_k": torch.zeros(1),
    }
    if top_k is not None:
        ret_dict["top_k"] = ml_utils.top_k_acc(
            ps[idx], labels[idx], top_k, as_tensor=True
        )
    return ret_dict

class DecayScheduler(_LRScheduler):
    """
    Code adapted from https://kikaben.com/transformers-training-details/
    to have a more flexible decay rate and relationship between
    warmup steps and the maximum learning rate.
    """
    @staticmethod
    def calc_lr(step,
            warmup_steps,
            max_lr=0.005,
            min_lr=1e-7,
            decay_exp=0.5,
        ):
        """
        Args:
            warmup_steps: int
            min_lr: float
                sets a lower bound on the learning rate. the lr will
                never drop below this value
            max_lr: float
                the maximum learning rate. This learning rate will
                be returned at the end of the warmup.
            decay_exp: float
                an exponent dictating the rate of decay of the learning
                rate following the warmup.
        """
        scale = max_lr * warmup_steps**(decay_exp)
        warmup = scale * step / warmup_steps**(1+decay_exp)
        reg = np.maximum(scale*step**(-decay_exp), min_lr)
        return np.minimum(reg, warmup)

    def __init__(self, 
                 optimizer,
                 warmup_steps: int=100,
                 last_epoch: int=-1,
                 verbose: bool=False,
                 min_lr: float=1e-10,
                 lr: float=1,
                 lr_decay_exp=0.25,
                 *args, **kwargs) -> None:
        """
        Args:
            warmup_steps: int
            min_lr: float
                sets a lower bound on the learning rate. the lr will
                never drop below this value
            lr: float
                the maximum learning rate. This learning rate will
                be returned at the end of the warmup.
            lr_decay_exp: float
                an exponent dictating the rate of decay of the learning
                rate following the warmup.
        """
        self.max_lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_exp = lr_decay_exp
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = DecayScheduler.calc_lr(
            self._step_count,
            warmup_steps=self.warmup_steps,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            decay_exp=self.lr_decay_exp,
        )
        return [float(lr)] * self.num_param_groups



if __name__=="__main__":
    #def blotch(idxs, sep_idx, blotch_p=0.4, indy_blotching=False):
    sep_idx = 0
    blotch_p = 1
    n_samples = 5000
    slen = 9
    allow_contig = True

    idxs = torch.randint(2,9, size=(4,slen))
    idxs[:2,:-1:2] = sep_idx
    idxs[2:,1:-1:2] = sep_idx
    print("Idxs:", idxs)

    for tok_p in [i/10. for i in range(11)]:
        print("tok_p:", tok_p)
        bmask = get_tok_mask(
            src=idxs,
            tok_p=tok_p,
            sep_idx=sep_idx,
        )
        for row in range(idxs.shape[0]):
            print()
            print("Unblotched:", idxs[row])
            print("Mask:", bmask[row])
            print("Blotched:", idxs[row][~bmask[row]])
        print()

    #blotches = torch.ones(len(idxs))*torch.FloatTensor([
    #    0.1, 0.3, 0.6, 0.9
    #])
    #n_seps = torch.max(torch.sum(idxs==sep_idx, dim=1))
    #for i in range(n_seps):
    #    bmask = get_blotch_mask(
    #        idxs,
    #        sep_idx=sep_idx,
    #        #blotch_p=blotches,
    #        allow_contig=allow_contig,
    #        step_idx = i
    #    )
    #    print("bmask:", bmask)
    #    print("blotch:", idxs[bmask])

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




