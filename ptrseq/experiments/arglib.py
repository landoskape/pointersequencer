from ..utils import argbool


def add_standard_training_parameters(parser):
    """arguments for defining the network type, dataset, optimizer, and other metaparameters"""
    parser.add_argument("--learning_mode", type=str, default="reinforce", help="which learning mode to use (default=reinforce)")
    parser.add_argument("--optimizer", type=str, default="Adam", help="what optimizer to train with (default=Adam)")
    parser.add_argument("--train_epochs", type=int, default=2000, help="how many epochs to train the networks on")
    parser.add_argument("--test_epochs", type=int, default=100, help="how many epochs to train the networks on")
    parser.add_argument("--replicates", type=int, default=2, help="how many replicates of each network to train")
    parser.add_argument("--silent", type=argbool, default=False, help="whether or not to print training progress (default=False)")
    parser.add_argument("--save_loss", default=False, action="store_true", help="save loss during training (default=False)")
    parser.add_argument("--save_reward", default=False, action="store_true", help="save reward during training (default=False)")
    return parser


def add_network_training_metaparameters(parser):
    """arguments for determining default network & training metaparameters"""
    parser.add_argument("--lr", type=float, default=1e-4, help="default learning rate (default=1e-4)")
    parser.add_argument("--wd", type=float, default=0, help="default weight decay (default=0)")
    parser.add_argument("--reward_gamma", type=float, default=1.0, help="default gamma for reward processing (default=1.0)")
    parser.add_argument(
        "--train_temperature",
        type=float,
        default=3.0,
        help="temperature for training (default=3.0, used for initial_value of scheduler if not provided, or overwritten by it if provided)",
    )
    parser.add_argument("--thompson", type=argbool, default=True, help="whether to use Thompson sampling during training (default=True)")
    parser.add_argument("--baseline", type=argbool, default=True, help="whether to use a baseline correction during training (default=True)")
    parser.add_conditional(
        "baseline",
        True,
        "--bl_temperature",
        type=float,
        default=1.0,
        help="temperature for baseline networks during training (default=1.0)",
    )
    parser.add_conditional(
        "baseline",
        True,
        "--bl_thompson",
        type=argbool,
        default=False,
        help="whether to use Thompson sampling for baseline networks (default=False)",
    )
    parser.add_conditional(
        "baseline",
        True,
        "--bl_significance",
        type=float,
        default=0.05,
        help="significance level for updating baseline networks (default=0.05)",
    )
    parser.add_conditional(
        "baseline",
        True,
        "--bl_batch_size",
        type=int,
        default=256,
        help="batch size for baseline networks (default=256)",
    )
    parser.add_conditional(
        "baseline",
        True,
        "--bl_frequency",
        type=int,
        default=10,
        help="how many epochs to wait before checking baseline improvement (default=10)",
    )
    return parser


def add_scheduling_parameters(parser, name="lr"):
    """arguments for scheduling a value based on the name given"""
    # check if it's a permitted parameter (based on how schedulers are used in train.train)
    permitted_parameters = ["lr", "train_temperature"]
    if name not in permitted_parameters:
        raise ValueError(f"parameter '{name}' not in permitted parameters: {permitted_parameters}")

    # helper function to create parameter names (used many times in parser.add_conditional)
    _prm = lambda prm: f"--{name}_{prm}"
    _dest = lambda dest: f"{name}_{dest}"
    use_name = _prm("use_scheduler")
    use_dest = _dest("use_scheduler")

    # add "{name}_use_scheduler" to determine whether to use a scheduler for this parameter
    parser.add_argument(use_name, type=argbool, default=False, help=f"whether to use a scheduler for {name} (default=False)")

    # add conditional argument for determining which scheduler to use and initial value / negative_clip (required by all)
    parser.add_conditional(use_dest, True, _prm("scheduler"), type=str, default="constant", help=f"which scheduler to use (default=constant)")
    parser.add_conditional(use_dest, True, _prm("initial_value"), type=float, default=None, help=f"initial value for the Scheduler (default=None)")
    parser.add_conditional(use_dest, True, _prm("negative_clip"), type=argbool, default=True, help=f"ignore negative epochs (default=True)")

    # add scheduler-specific conditional parameters
    parser.add_conditional(
        _dest("scheduler"),
        "step",
        _prm("step_size"),
        type=int,
        required=True,
        help=f"step size for the StepScheduler (required)",
    )
    parser.add_conditional(
        _dest("scheduler"),
        lambda val: val in ["step", "exp", "expbase"],
        _prm("gamma"),
        type=float,
        required=True,
        help=f"gamma for the scheduler (required)",
    )
    parser.add_conditional(
        _dest("scheduler"),
        lambda val: val in ["expbase", "linear"],
        _prm("final_value"),
        type=float,
        required=True,
        help=f"final_value for the scheduler (required)",
    )
    parser.add_conditional(
        _dest("scheduler"),
        "linear",
        _prm("total_epochs"),
        type=int,
        required=True,
        help=f"total_epochs for the (required)",
    )
    return parser


def _add_transformer_parameters(parser, name, num_heads=8, kqnorm=True, expansion=1, kqv_bias=False, mlp_bias=True, residual=True):
    """add conditional parameters for a transformer layer"""
    _prm = lambda prm: f"--{name}_{prm}"
    _dest = lambda dest: f"{name}_{dest}"
    _in_both = lambda val: val in ["attention", "transformer"]
    parser.add_conditional(
        _dest("method"), _in_both, _prm("num_heads"), type=int, default=num_heads, help=f"the number of heads in {name} layers (default={num_heads})"
    )
    parser.add_conditional(
        _dest("method"), _in_both, _prm("kqnorm"), type=argbool, default=kqnorm, help=f"whether to use kqnorm in the {name} (default={kqnorm})"
    )
    parser.add_conditional(
        _dest("method"),
        "transformer",
        _prm("expansion"),
        type=int,
        default=expansion,
        help=f"the expansion of the FF layers in the {name} (default={expansion})",
    )
    parser.add_conditional(
        _dest("method"),
        _in_both,
        _prm("kqv_bias"),
        type=argbool,
        default=kqv_bias,
        help=f"whether to use bias in the attention kqv layers (default={kqv_bias})",
    )
    parser.add_conditional(
        _dest("method"),
        "transformer",
        _prm("mlp_bias"),
        type=argbool,
        default=mlp_bias,
        help=f"use bias in the MLP part of the {name} (default={mlp_bias})",
    )
    parser.add_conditional(
        _dest("method"),
        "attention",
        _prm("residual"),
        type=argbool,
        default=residual,
        help=f"use residual connections in the attentional {name} (default={residual})",
    )
    return parser


def add_pointernet_parameters(parser, no_encoder=False, no_decoder=False, no_pointer=False):
    """
    arguments for the PointerNet including conditionals for encoder/decoder/pointer layer methods

    adds arguments for the embedding, encoder, decoder, and pointer layers in a PointerNet
    the encoder, decoder, and pointer layers have a "method" which is a string that determines
    which kind of layer to use. The conditional arguments associated with it are added.

    Sometimes all conditional arguments need to be added regardless, so it's possible to turn
    off the conditional adding by using no_{encoder,decoder,pointer}=True.
    """
    parser.add_argument("--embedding_dim", type=int, default=128, help="the dimensions of the embedding (default=128)")
    parser.add_argument("--embedding_bias", type=argbool, default=True, help="whether to use embedding_bias (default=True)")

    parser.add_argument("--num_encoding_layers", type=int, default=1, help="the number of encoding layers in the PointerNet (default=1)")
    parser.add_argument("--encoder_method", type=str, default="transformer", help="PointerNet encoding layer method (default='transformer')")
    if not no_encoder:
        parser = _add_transformer_parameters(parser, "encoder")

    parser.add_argument("--decoder_method", type=str, default="transformer", help="PointerNet decoding layer method (default='transformer')")
    if not no_decoder:
        parser = _add_transformer_parameters(parser, "decoder")
        parser.add_conditional(
            "decoder_method",
            "gru",
            "--decoder_gru_bias",
            type=argbool,
            default=True,
            help="whether to use bias in the gru decoder method (default=True)",
        )

    parser.add_argument("--pointer_method", type=str, default="standard", help="PointerNet pointer layer method (default='standard')")
    if not no_pointer:
        parser = _add_transformer_parameters(parser, "pointer")
        _bias_required = lambda val: val in ["standard", "dot", "dot_noln"]
        parser.add_conditional(
            "pointer_method",
            _bias_required,
            "--pointer_bias",
            type=argbool,
            default=False,
            help="whether to use bias in pointer projection layers (default=False)",
        )
    return parser


def add_pointer_layer_parameters(parser, num_heads=8, kqnorm=True, expansion=1, bias=False, kqv_bias=False, mlp_bias=True, residual=True):
    """arguments for all possible pointer layer in a PointerNet"""
    parser.add_argument("--pointer_num_heads", type=int, default=num_heads, help=f"number of heads in pointer layers (default={num_heads})")
    parser.add_argument("--pointer_kqnorm", type=argbool, default=kqnorm, help=f"use kqnorm in pointerlayer (default={kqnorm})")
    parser.add_argument("--pointer_expansion", type=int, default=expansion, help=f"expansion of the FF layers in pointerlayer (default={expansion})")
    parser.add_argument("--pointer_bias", type=argbool, default=bias, help=f"use bias in pointerlayers (default={bias})")
    parser.add_argument("--pointer_kqv_bias", type=argbool, default=kqv_bias, help=f"use attention bias in pointer layers (default={kqv_bias})")
    parser.add_argument("--pointer_mlp_bias", type=argbool, default=mlp_bias, help=f"use MLP bias pointer layers (default={mlp_bias})")
    parser.add_argument("--pointer_residual", type=argbool, default=residual, help=f"use residuals in attentional pointerlayers (default={residual})")
    return parser


def add_checkpointing(parser):
    """arguments for managing checkpointing when training networks"""
    parser.add_argument("--use_prev_ckpts", default=False, action="store_true", help="pick up training off previous checkpoint (default=False)")
    parser.add_argument("--save_ckpts", default=False, action="store_true", help="save checkpoints of models (default=False)")
    parser.add_conditional(
        "save_ckpts",
        True,
        "--uniq_ckpts",
        default=False,
        action="store_true",
        help="save unique checkpoints of models each epoch (default=False)",
    )
    parser.add_conditional(
        "save_ckpts",
        True,
        "--freq_ckpts",
        default=1,
        type=int,
        help="frequency (by epoch) to save checkpoints of models (default=1)",
    )
    parser.add_argument("--use_wandb", default=False, action="store_true", help="log experiment to WandB (default=False)")
    return parser


def add_dataset_parameters(parser):
    """add generic dataset parameters"""
    parser.add_argument("--task", type=str, required=True, help="which task to use (the dataset to load), required")
    parser.add_argument("--batch_size", type=int, default=128, help="what batch size to pass to DataLoader")
    parser.add_argument("--threads", type=int, default=1, help="the number of threads to use for generating batches (default=1)")
    parser.add_argument("--ignore_index", type=int, default=-100, help="the index to ignore in the loss function (default=-100)")

    # conditional parameters for each task
    is_dominoe_task = lambda x: x in ["dominoe_sequencer", "dominoe_sorter"]
    parser.add_conditional(
        "task",
        is_dominoe_task,
        "--highest_dominoe",
        type=int,
        default=9,
        help="the highest dominoe value (default=9)",
    )
    parser.add_conditional(
        "task",
        is_dominoe_task,
        "--train_fraction",
        type=float,
        default=0.9,
        help="the fraction of dominoes to train with (default=0.9)",
    )
    parser.add_conditional(
        "task",
        is_dominoe_task,
        "--hand_size",
        type=int,
        default=12,
        help="the number of dominoes in the hand (default=12)",
    )
    parser.add_conditional(
        "task",
        is_dominoe_task,
        "--randomize_direction",
        type=argbool,
        default=True,
        help="randomize the direction of the dominoes (default=True)",
    )
    parser.add_conditional(
        "task",
        "dominoe_sequencer",
        "--value_method",
        type=str,
        default="length",
        help="how to calculate the value of a sequence (default=length)",
    )
    parser.add_conditional(
        "task",
        "dominoe_sequencer",
        "--value_multiplier",
        type=float,
        default=1.0,
        help="how to scale the value of a sequence (default=1.0)",
    )
    parser.add_conditional(
        "task",
        "dominoe_sorter",
        "--allow_mistakes",
        type=argbool,
        default=False,
        help="allow mistakes in the sorting task (default=False)",
    )
    parser.add_conditional(
        "task",
        "tsp",
        "--num_cities",
        type=int,
        default=10,
        help="the number of cities in the TSP (default=10)",
    )
    parser.add_conditional(
        "task",
        "tsp",
        "--coord_dims",
        type=int,
        default=2,
        help="the number of dimensions for the coordinates (default=2)",
    )

    # add curriculum related arguments
    parser.add_argument("--use_curriculum", type=argbool, default=False, help="use curriculum training (default=False)")
    parser.add_conditional(
        "use_curriculum",
        True,
        "--curriculum_epochs",
        type=int,
        nargs="*",
        default=[1000, 1000],
        help="how many epochs to train with curriculum (default=[1000, 1000])",
    )
    return parser
