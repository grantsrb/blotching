# Semantic Blotching

## Using this repo
After cloning, you will first need to initialize the `ml_utils`
submodule. You can do this with the following commands at the terminal:

    $ git clone https://github.com/grantsrb/semantic_blotching.git
    $ cd semantic_blotching
    $ cd ml_utils
    $ git submodule init
    $ git submodule update
    $ cd ../

Next, you will need to make sure you have all necessary pacakges
installed.

Lastly, you can run a training by creating a hyperparameters.json and
then running the following command:

    # python main.py hyperparameters.json

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
        optionally start from an existing model checkpoint. this should
        be the full path to the checkpoint that you would like to use.
        This will not use the hyperparameters from the argued checkpt,
        just the model weights.

    "star": bool
        if true, will use model to sample new data that is selected
        for correctness and brevity
    "pre_epochs": int
        the number of epochs to wait before sampling new data
    "n_runner_procs": int
        the number of parallel processes to use for collecting new
        data using the model
    "collection_size": int
        the number of new samples to attempt when collecting new data.

    "max_val_loops": int
        enforces a limit on the number of validation iterations. This
        is useful for speeding up trainings
    "n_train_loops": int
        the number of loops per epoch. this is useful if you want to
        validate more often.
    "checkpt_mod": int or None
        during training, the model will be saved every `checkpt_mod`
        iterations

    "n_epochs": int
        the total number of training iterations
    "batch_size": int
        the size of batches for stochastic gradient descent
    "lr": float
        the learning rate
    "l2": float
        the l2 norm regularization. aka weight decay
    "seq_len": int
        the data sequence length to use. for causal modeling, `seq_len`
        refers to the sequence length post compression, so the model will
        compress `cmp_len` tokens and then predict `seq_len` tokens.
        if doing rmb_only, `cmp_len` is ignored
    "blotch_p": float
        the blotch probability. 0 means no blotching. blotching is
        effectively contiguous dropout. It is kept to complete
        sequences, however, rather than fully random.
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

    
