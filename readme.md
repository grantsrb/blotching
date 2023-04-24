# Semantic Blotching

## Using this repo
After cloning, you will first need to initialize the `ml_utils`
submodule. You can do this with the following commands at the terminal:

    $ cd ml_utils
    $ git submodule init
    $ git submodule update

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

    "dataset": str
        a string of the huggingface datasets dataset you would like to
        use. Currently only support "openwebtext" and "glue"
    "n_data_procs": int
        the number of parallel processes to use for the initial
        encoding of the data.

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

    
