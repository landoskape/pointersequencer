from natsort import natsorted
import torch

from .wrangling import check_similarity


def make_checkpoint_path(path_ckpt, epoch, uniq_ckpts, prefix=None):
    """make checkpoint path unique for each epoch if requested"""
    # if unique, add the epoch to the checkpoint path
    if uniq_ckpts:
        return path_ckpt / "checkpoints" / f"checkpoint_{prefix}_{epoch}.tar"

    # otherwise just save a single checkpoint file (with possible prefix)
    if prefix is not None:
        return path_ckpt / f"checkpoint_{prefix}.tar"

    return path_ckpt / "checkpoint.tar"


def get_checkpoint_path(path_ckpt, epoch=None, prefix=None):
    """get checkpoint path for loading or saving"""
    # if epoch is provided, search for the specific checkpoint requested
    if epoch is not None or prefix is not None:
        # this way the user is asking for a specific checkpoint
        path = make_checkpoint_path(path_ckpt, epoch, epoch is not None, prefix=prefix)
        if path.exists():
            return path

    # search for latest unique checkpoint
    prev_checkpoints = list((path_ckpt / "checkpoints").glob("checkpoint_*.tar"))
    single_checkpoint = path_ckpt / "checkpoint.tar"

    # if unique checkpoints found, return the latest one (with a warning if non-unique checkpoint also exists)
    if prev_checkpoints:
        if single_checkpoint.exists():
            print(f"WARNING: found both unique and non-unique checkpoints in {path_ckpt}, using latest unique checkpoint.")
        return natsorted(prev_checkpoints)[-1]

    # if no unique checkpoints found, return the single checkpoint if it exists
    if single_checkpoint.exists():
        return single_checkpoint

    # otherwise return None to indicate that no checkpoints were found
    return None


def save_checkpoint(nets, optimizers, results, parameters, epoch, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = dict(results=results) | multi_model_ckpt | multi_optimizer_ckpt | dict(epoch=epoch) | dict(parameters=parameters)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    torch.save(checkpoint, path)


def _update_results(results, ckpt_results, num_completed):
    """
    Helper method for updating results with checkpoint results.
    """
    for key in results:
        if isinstance(results[key], dict):
            _update_results(results[key], ckpt_results[key], num_completed)
        elif isinstance(results[key], torch.Tensor):
            results[key][:num_completed] = ckpt_results[key][:num_completed]
        else:
            print(f"skipping {key} in checkpoint results update (not a tensor)")


def last_checkpoint_epoch(path):
    """Method for getting the last epoch from a checkpoint."""
    checkpoint = torch.load(path)
    return checkpoint["epoch"] + 1


def load_checkpoint(nets, optimizers, results, parameters, device, path):
    """
    Method for loading presaved checkpoint during training.
    TODO: device handling for passing between gpu/cpu
    """

    print(f"loading from latest checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    ckpt_results = checkpoint["results"]
    ckpt_parameters = checkpoint["parameters"]
    if ckpt_parameters["num_epochs"] > parameters["num_epochs"]:
        raise ValueError(f"checkpoint num_epochs ({ckpt_parameters['num_epochs']}) is greater than current num_epochs ({parameters['num_epochs']})")

    # check if results and ckpt results have same structure
    check_similarity(results, ckpt_results, name1="results", name2="ckpt_results", compare_shapes=False, compare_dims=True)

    # update results if they are consistent
    num_completed = checkpoint["epoch"] + 1
    _update_results(results, ckpt_results, num_completed)

    # get ids for each net and optimizer
    net_ids = natsorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = natsorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])

    # check equivalence
    msg = "nets and optimizers cannot be matched up from checkpoint"
    assert all([oi.split("_")[-1] == ni.split("_")[-1] for oi, ni in zip(opt_ids, net_ids)]), msg

    # load state dicts for nets and optimizers
    for net, net_id in zip(nets, net_ids):
        net.load_state_dict(checkpoint.pop(net_id))
    for opt, opt_id in zip(optimizers, opt_ids):
        opt.load_state_dict(checkpoint.pop(opt_id))

    # put nets on correct device
    for net in nets:
        net.to(device)

    return nets, optimizers, results, num_completed
