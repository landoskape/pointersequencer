import torch
from natsort import natsorted


def save_checkpoint(nets, optimizers, results, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = results | multi_model_ckpt | multi_optimizer_ckpt
    torch.save(checkpoint, path)


def load_checkpoints(nets, optimizers, device, path):
    """
    Method for loading presaved checkpoint during training.
    TODO: device handling for passing between gpu/cpu
    """

    prev_checkpoints = list(path.glob("checkpoint_*"))
    latest_checkpoint = natsorted(prev_checkpoints)[-1] if prev_checkpoints else path / "checkpoint.tar"
    print(f"loading from latest checkpoint: {latest_checkpoint}")
    if device == "cpu":
        checkpoint = torch.load(latest_checkpoint, map_location=device)
    elif device == "cuda":
        checkpoint = torch.load(latest_checkpoint)
    elif isinstance(device, int):
        map_location = {"cuda:0": f"cuda:{device}"}
        checkpoint = torch.load(latest_checkpoint, map_location=map_location)

    net_ids = natsorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = natsorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])

    msg = "nets and optimizers cannot be matched up from checkpoint"
    assert all([oi.split("_")[-1] == ni.split("_")[-1] for oi, ni in zip(opt_ids, net_ids)]), msg

    # load state dicts for nets and optimizers
    for net, net_id in zip(nets, net_ids):
        net.load_state_dict(checkpoint.pop(net_id))
    for opt, opt_id in zip(optimizers, opt_ids):
        opt.load_state_dict(checkpoint.pop(opt_id))

    if device == "cuda":
        for net in nets:
            net.to(device)

    return nets, optimizers, checkpoint
