# Chain of Shortcuts

See a writeup of this project [here.](https://github.com/grantsrb/blotching/blob/master/Grant-2023-ChainOfShortcuts.pdf)

Another paper called [GPT Can Solve Mathematical Problems Without a Calculator](https://arxiv.org/abs/2309.03241) came out in September that largely does what we were moving towards. Overall, our direction doesn't seem that promising as a longterm focus, thus this project is largely dead 😢

## Hyperparameters

    "exp_name": str
        the name of the folder in which the hyperparameter search will
        be saved to. This is different than the path. If you would like
        to save the experiment to a different folder than the one in
        which you run `main.py`, use the hyperparemter called `save_root`
    "save_root": str
        this value is prepended to the exp_name when creating the save
        folder for the hyperparameter search.
    "data_root": str
        the path to where the processed datasets are saved
    "multi_gpu": bool
        if true, the script will try a data parallel approach, splitting
        the batches accross multiple gpus
    "model_parallel": bool
        if true, the script will use Huggingface's auto device map
        feature.
    "torch_dtype": str
        the floating point precision to use. for example: "float32"
    "seed": int
        the random seed for all stochastic processes

    "init_checkpt": str or None
        optionally start from an existing model checkpoint. will also
        load the checkpoint's training and validation data. the arguement
        should be the full path to the checkpoint that you would like to
        use.  This will not use the hyperparameters from the argued
        checkpt, just the model weights and data.

    "max_samples": int or null
        the maximum number of data samples in the data cache. if null
        defaults to `init_samples`. `max_samples` or `init_samples`
        must be not null.
    "init_samples": int or null
        the initialization quantity of data samples. If null, defaults
        to `max_samples`. `max_samples` or `init_samples` must be not
        null.
    "val_samples": int
        the number of validation samples
    "val_mod": int
        the number of training loops to perform per validation. i.e.
        a value of 3 means there will be 3 training updates for every
        validation

    "axe_loops": int
        the number of loops to perform in which the samples are axed,
        meaning they permanently lose a segment if the model got the
        problem correct and the relative loss difference is less than
        some tolerance
    "abs_axe_tol": bool
        if true, will use absolute value of loss difference when axing
        samples. Otherwise will use the `axe_tol` as a tolerance for
        proportional change
    "axe_tol": float
        the tolerence of the loss difference to decide when to keep
        axings
    "axe_comp_steps": int
        the number of steps to compare when axing
    "axe_use_ppl": bool
        if true, will use perplexity instead of entropy for evaluating
        axing samples.
    "axe_blotch_p": float or None
        the amount of blotching for btoken models during the axing phase.
        If None, defaults to `bootstrap_blotch_p`. Only applies when
        using btoken models.

    "aug_loops": int
        if greater than 0, will use the model to augment samples in the
        training data. Augmentations are only kept when the final answer
        is correct and the total answer is shorter than the original.
        The value of this parameter indicates the number of augmentation
        loops to perform after each epoch. Each loop attempts to augment
        `val_batch_size` uniformly sampled data points from the existing
        data.

    "star_loops": int
        if greater than 0, will use the model to collect new samples for
        the training data. STaR samples are only kept when the final
        answer is correct and the total answer is shorter than the
        original. The value of `star_loops` indicates the number of
        bootstrap loops to perform after each epoch. Each loop attempts
        to collect `val_batch_size` newly sampled problems
    "star_force_short": bool
        if true, star bootstrapped samples will only be collected if they
        are shorter than the ground truth solution. Otherwise, only
        considers the accuracy of the final solution for collection.

    "pre_epochs": int
        the number of epochs to wait before sampling new data
    "in_place": bool
        if true, the augmentations will replace the existing examples.
        if false, the augmentations will be added to the existing
        dataset.
    "aug_mod": int
        a modulus to determine the frequency of data manipulations.
        manipulations are augmentations, axings, or star bootstraps

    "max_val_loops": int
        enforces a limit on the number of validation iterations. This
        is useful for speeding up trainings
    "n_train_loops": int
        the number of loops per epoch. this is useful if you want to
        validate more often.
    "checkpt_mod": int or None
        during training, the model will be saved every `checkpt_mod`
        iterations

    "max_num": int
        the maximum number available for sampling the initial
        problem (inclusive)
    "max_ents": int
        maximum entities for the starting problem. If using
        parentheticals, a parenthetical counts as one entity,
        parentheticals are recursively samples with max_ents-1
        max entities.
    "p_mult": float [0,1]
        the probability of sampling a multiplication sign for
        the starting problem
    "max_mult_num": int
        the maximum values that can be included in a multiplication
        operation
    "p_paren": float [0,1]
        the probability of sampling a parenthetical.
        Parentheticals are sampled the same way as the initial
        problem but with max entities equal to max_ents-1
    "space_mults": bool
        if true, will not allow more than two numbers to be
        multiplied together
    "zipf_order": float
        the exponent of a zipfian distribution by which to
        sample each entity.
    "p_ent": float [0,1]
        the probability of samping each entity beyond the first
        two. There will always be at least two entities, but
        beyond that, each entity has a p_ent chance of being
        sampled at all. In the case that each entity fails to
        be sampled, the resulting problem has fewer entities.
        A value of 1 means each entity is guaranteed to be
        sampled, a value of 0 is equivalent to setting the
        max_ents to 2.

    "n_epochs": int
        the total number of training iterations
    "batch_size": int
        the size of batches for stochastic gradient descent
    "lr": float
        the learning rate. also sets the maximum learning rate in
        schedulers with decay and warmup.
    "l2": float
        the l2 norm regularization. aka weight decay
    "seq_len": int
        the data sequence length to use. for causal modeling, `seq_len`
        refers to the sequence length post compression, so the model will
        compress `cmp_len` tokens and then predict `seq_len` tokens.
        if doing rmb_only, `cmp_len` is ignored
    "tok_drop_p": float
        the probability of randomly dropping a token in the sequence.
        0 means no token dropping.
    "blotch_p": float
        the blotch probability. 0 means no blotching. blotching is
        effectively contiguous dropout. It is kept to complete
        sequences, however, rather than fully random. If using a
        BlotchTokenModel, use `blotch_p_min` and `blotch_p_max` instead.
    "blotch_p_min": float
        sets the minimum amount of blotching for a model.
    "blotch_p_max": float
        sets the maximum amount of blotching for a model.
    "n_btokens": int or None
        the number of blotch tokens. This is effectively a
        granularity parameter for blotch values. If None, will
        default to blotch_p increments of 0.1 on the difference
        of bp_max and bp_min. For example, if bp_max-bp_min is
        0.4, then there will be 0.4/0.1 = 4 tokens
    "bootstrap_blotch_p": float
        the amount of blotching to use for the bootstrapping phases.
        only applies to BlotchTokenModel type.
    "contig_blotches": bool
        if true, will allow contiguous blotches. If false, will
        separate blotch segments by at least one semantic step

    "d_model": int
        the embedding and hidden state dimensionality of the model
    "n_heads": int
        the number of attention heads for transformer models
    "h_mult": int
        a multiplicative term that determines the hidden dimensionality
        for all multilayer FFNs in the model.
    "n_layers": int
        the number of transformer layers or consecutive lstms used in
        the model.

    "model_type": str
        the class name of the model architecture
    "hf_model_type": str ("gpt2", "gptj", "llama")
        if using huggingface model types, can argue the string name
        of the model type you would like to use.
    "pretrained": bool
        if true, will ignore model configuration parameters and use
        the pretrained version of the huggingface model type. only
        applies when using hf model types.

    "digit_embs": bool
        if true, all numbers consist of individual digit embeddings.
        false is not currently implemented.
    "posenc_type": str
        the type of positional encodings. As of now, valid arguments
        are "RandPositionalEncoding" and "SinPositionalEncoding"
    "max_posencs": int
        the maximum number of positional encodings. In the case of
        SinPositionalEncoding, this is the maximum possible length
        of a sequence. In the case of the RandPositionalEncodings,
        this is the number of positional encodings to sample from.
    "posenc_drop_p": float
        the dropout specific to positional encodings.
    "learnable_posencs": bool
        if true, gradients are backpropagated into the positional
        encodings
    "pad_pos_skip": bool
        if this is true, the positional encodings will be added in
        such a way that they skip tokens based off true values of the
        `pad_mask`. i.e. the blotched tokens and padded tokens are
        removed from determining positional indexes.

    "incl_intl_prob": bool
        if true, will include the initial problem in the loss. Otherwise
        will exclude the initial problem from the causal modeling.
        It is impossible to predict the initial problem, as such it is
        not important to train to predict the initial problem.

    "n_grad_loops": int
        the number of backprop loops to perform before performing an
        optimizer step. the loss is divided by this quantity so as to
        be equivalent to stepping the optimizer once per iteration with
        a batch size of batch_size*n_grad_loops

    "label_smoothing": float
        a value from 0-1 to apply label smoothing to the loss
    "plateau_scheduler": bool
        if true, the learning rate is scheduled to decrease on performance
        plateaus. Otherwise uses a warm up (see `warmup_steps`) and
        then proceeds to decay to a minimum learning rate set by
        `min_lr`
    "warmup_steps": int
        the number of training steps (not epochs) to warmup the learning
        rate from 0 to max(lr*1/sqrt(step num), `min_lr`)
    "min_lr": float
        the minimum possible learning rate
    "lr_decay_exp": float
        the exponent dictating the rate of decay of the learning rate
        after the warmup steps.

    "blotch_spacing": str, one of ["random", "equal"]
        optionally pick how the blotching occurs with the following
        options:

        "random": blotching is random according to the blotching probability
        "equal": the blotching is semi-deterministicly
                    distributed relatively equally amongst the possible
                    segments.  The number of blotched segments is decided
                    by `blotch_p * <num possible segments>` where the
                    rounding direction is decided randomly using a
                    probability equal to the remaining fraction. So,
                    there's still some stochasticity but it's lower
                    variance and deterministic in many cases.
        Not Implemented:
            "random_equal": the blotching is mostly spaced equally except
                that each equally spaced blotch segment has an equal
                probability of being located at the segment before or after
                the equally spaced segement. i.e. there is 2/3 probability
                that instead of blotching the segment chosen by equal
                spacing, we instead blotch the segment immediately before
                or after with equal probability.
        
        
