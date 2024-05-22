from natsort import natsorted
import torch

from .wrangling import check_similarity


def make_checkpoint_path(path_ckpt, epoch, uniq_ckpts):
    """make checkpoint path unique for each epoch if requested"""
    # if unique, add the epoch to the checkpoint path
    if uniq_ckpts:
        return path_ckpt / "checkpoints" / f"checkpoint_{epoch}.tar"

    # otherwise just save a single checkpoint file
    return path_ckpt / "checkpoint.tar"


def save_checkpoint(nets, optimizers, results, parameters, epoch, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = results | multi_model_ckpt | multi_optimizer_ckpt | dict(epoch=epoch) | dict(parameters=parameters)
    torch.save(checkpoint, path)


def _update_results(results, ckpt_results, num_completed):
    """
    Helper method for updating results with checkpoint results.
    """
    for key in results:
        if isinstance(results[key], dict):
            _update_results(results[key], ckpt_results[key])
        else:
            results[key][:num_completed] = ckpt_results[key][:num_completed]


def load_checkpoint(nets, optimizers, results, device, path):
    """
    Method for loading presaved checkpoint during training.
    TODO: device handling for passing between gpu/cpu
    """
    # get latest checkpoint (first check for unique checkpoints, then check for checkpoint.tar)
    prev_checkpoints = list((path / "checkpoints").glob("checkpoint_*"))
    latest_checkpoint = natsorted(prev_checkpoints)[-1] if prev_checkpoints else path / "checkpoint.tar"

    print(f"loading from latest checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    ckpt_results = checkpoint["results"]

    # check if results and ckpt results have same structure
    check_similarity(results, ckpt_results, name1="results", name2="ckpt_results", compare_shapes=False, compare_dims=True)

    # update results if they are consistent
    _update_results(results, ckpt_results, checkpoint["epoch"])

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

    return nets, optimizers, results, checkpoint["epoch"]
