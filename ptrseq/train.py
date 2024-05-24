from tqdm import tqdm
import torch
from .networks.net_utils import forward_batch
from .networks.baseline import make_baseline_nets, check_baseline_updates
from .utils import train_nets, test_nets, get_scheduler
from .utils import make_checkpoint_path, get_checkpoint_path, save_checkpoint, load_checkpoint


@train_nets
def train(nets, optimizers, dataset, **parameters):
    """a generic training function for pointer networks"""
    num_nets = len(nets)
    assert num_nets == len(optimizers), "Number of networks and optimizers must match"

    # get some key training parameters
    starting_epoch = 0
    num_epochs = parameters.get("num_epochs")
    device = parameters.get("device")
    verbose = parameters.get("verbose", True)
    max_possible_output = parameters.get("max_possible_output")  # this is the maximum number of outputs ever
    learning_mode = parameters.get("learning_mode")
    temperature = parameters.get("temperature", get_scheduler("constant", build=True, initial_value=1.0))
    thompson = parameters.get("thompson", True)
    baseline = parameters.get("baseline", True) and learning_mode == "reinforce"

    # create gamma transform for processing reward if not provided in parameters
    if learning_mode == "reinforce":
        gamma = parameters.get("gamma")
        gamma_transform = dataset.create_gamma_transform(max_possible_output, gamma, device=device)

    # process the learning_mode and save conditions
    get_loss = learning_mode == "supervised" or parameters.get("save_loss", False)
    get_reward = learning_mode == "reinforce" or parameters.get("save_reward", False)

    # create a results dictionary for storing training data
    results = dict()

    # create some variables for storing data related to supervised loss
    if get_loss:
        results["loss"] = torch.zeros(num_epochs, num_nets, device="cpu")

    # create some variables for storing data related to rewards
    if get_reward:
        results["reward"] = torch.zeros(num_epochs, num_nets, device="cpu")

    # create dataset-specified variables for storing data
    results["dataset_variables"] = dataset.create_training_variables(num_nets, **parameters)

    # load checkpoints if required
    use_prev_ckpts = parameters.get("use_prev_ckpts", False)
    if use_prev_ckpts:
        path_ckpts = parameters.get("path_ckpts")  # required if use_prev_ckpts is True
        checkpoint_path = get_checkpoint_path(path_ckpts)
        if checkpoint_path is not None:
            nets, optimizers, results, starting_epoch = load_checkpoint(nets, optimizers, results, parameters, device, checkpoint_path)
            print("resuming training from checkpoint on epoch", starting_epoch)

    # handle checkpoint saving parameters
    save_ckpts = parameters.get("save_ckpts", False)
    if save_ckpts:
        uniq_ckpts = parameters.get("uniq_ckpts", False)
        freq_ckpts = parameters.get("freq_ckpts", 1)
        path_ckpts = parameters.get("path_ckpts")  # required if save_ckpts is True

    # prepare baseline networks if required
    if baseline:
        bl_temperature = parameters.get("bl_temperature", 1.0)
        bl_thompson = parameters.get("bl_thompson", False)
        bl_significance = parameters.get("bl_significance", 0.05)
        bl_batch_size = parameters.get("bl_batch_size", 256)
        bl_frequency = parameters.get("bl_frequency", 10)
        bl_parameters = parameters.copy()
        bl_parameters["batch_size"] = bl_batch_size  # update batch size for baseline reference batch
        bl_nets = make_baseline_nets(
            nets,
            dataset,
            batch_parameters=bl_parameters,
            significance=bl_significance,
            temperature=bl_temperature,
            thompson=bl_thompson,
        )

    # epoch loop
    epoch_loop = tqdm(range(starting_epoch, num_epochs), desc="Training Networks") if verbose else range(starting_epoch, num_epochs)
    for epoch in epoch_loop:
        # update scheduler values
        temperature.step(epoch=epoch)

        # generate a batch
        batch = dataset.generate_batch(**parameters)

        # zero gradients
        for opt in optimizers:
            opt.zero_grad()

        scores, choices = forward_batch(nets, batch, max_possible_output, temperature=temperature.get_value(), thompson=thompson)

        # get baseline choices if using them
        if baseline:
            with torch.no_grad():
                bl_choices = forward_batch(bl_nets, batch, max_possible_output, temperature=bl_temperature, thompson=bl_thompson)[1]

        # get loss
        if get_loss:
            loss = dataset.measure_loss(scores, batch["target"], check_divergence=True)

        # get reward
        if get_reward:
            rewards = [dataset.reward_function(choice, batch) for choice in choices]
            if baseline:
                with torch.no_grad():
                    bl_rewards = [dataset.reward_function(choice, batch) for choice in bl_choices]

        # backprop with supervised learning (usually using negative log likelihood loss)
        if learning_mode == "supervised":
            for l in loss:
                l.backward()

        # backprop with reinforcement learning (with the REINFORCE algorithm)
        if learning_mode == "reinforce":
            # get max output for this batch
            max_output = batch.get("max_output", max_possible_output)
            # get processed rewards, do backprop
            c_gamma_transform = gamma_transform[:max_output][:, :max_output]  # only use the part of gamma_transform that is needed
            _, J = dataset.process_rewards(
                rewards,
                scores,
                choices,
                c_gamma_transform,
                baseline_rewards=bl_rewards if baseline else None,
            )
            for j in J:
                j.backward()

        # update networks
        for opt in optimizers:
            opt.step()

        # update baseline networks if required
        if baseline and epoch % bl_frequency == 0:
            bl_nets = check_baseline_updates(nets, bl_nets)

        # save training data
        with torch.no_grad():
            if get_loss:
                for i in range(num_nets):
                    results["loss"][epoch, i] = loss[i].detach().cpu()

            if get_reward:
                for i in range(num_nets):
                    results["reward"][epoch, i] = torch.mean(torch.sum(rewards[i], dim=1)).detach().cpu()

            # save dataset-specific variables
            epoch_state = dict(
                epoch=epoch,
                batch=batch,
                scores=scores,
                choices=choices,
                loss=loss if get_loss else None,
                rewards=rewards if get_reward else None,
                gamma_transform=gamma_transform if learning_mode == "reinforce" else None,
                temperature=temperature.get_value(),
            )
            dataset.save_training_variables(results["dataset_variables"], epoch_state, **parameters)

        if save_ckpts and ((epoch % freq_ckpts == 0) or (epoch == (num_epochs - 1))):
            save_checkpoint(
                nets,
                optimizers,
                results,
                parameters,
                epoch,
                make_checkpoint_path(path_ckpts, epoch, uniq_ckpts),
            )

    # return training data
    return results


@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """a generic boiler plate function for testing and evaluating pointer networks"""
    num_nets = len(nets)

    # get some key training parameters
    num_epochs = parameters.get("num_epochs")
    verbose = parameters.get("verbose", False)
    max_possible_output = parameters.get("max_possible_output")  # this is the maximum number of outputs ever
    temperature = parameters.get("temperature", get_scheduler("constant", build=True, initial_value=1.0))
    thompson = parameters.get("thompson", False)

    # process save conditions
    get_loss = parameters.get("save_loss", False)
    get_reward = parameters.get("save_reward", False)
    get_target_reward = get_reward and parameters.get("return_target", False)

    # create some variables for storing data related to supervised loss
    if get_loss:
        track_loss = torch.zeros(num_epochs, num_nets, device="cpu")

    # create some variables for storing data related to rewards
    if get_reward:
        track_reward = torch.zeros(num_epochs, num_nets, device="cpu")
        track_reward_by_pos = torch.zeros(num_epochs, max_possible_output, num_nets, device="cpu")
        track_confidence = torch.zeros(num_epochs, max_possible_output, num_nets, device="cpu")
        if get_target_reward:
            batch_size = dataset.parameters(**parameters).get("batch_size", None)
            track_target_reward = torch.zeros(num_epochs, batch_size, device="cpu")
            track_network_reward = torch.zeros(num_epochs, batch_size, num_nets, device="cpu")

    # create dataset-specified variables for storing data
    dataset_variables = dataset.create_testing_variables(num_nets, **parameters)

    # epoch loop
    epoch_loop = tqdm(range(num_epochs), desc="Testing Networks") if verbose else range(num_epochs)
    for epoch in epoch_loop:
        # generate a batch
        batch = dataset.generate_batch(**parameters)

        scores, choices = forward_batch(nets, batch, max_possible_output, temperature=temperature.get_value(), thompson=thompson)

        # get loss
        if get_loss:
            loss = dataset.measure_loss(scores, batch["target"], check_divergence=True)

        # get reward
        if get_reward:
            rewards = [dataset.reward_function(choice, batch) for choice in choices]
            if get_target_reward:
                target_as_choice = dataset.target_as_choice(batch["target"])
                track_target_reward[epoch] = torch.sum(dataset.reward_function(target_as_choice, batch).detach().cpu(), dim=1)

        # save training data
        if get_loss:
            for i in range(num_nets):
                track_loss[epoch, i] = loss[i].detach().cpu()

        if get_reward:
            pretemp_scores = dataset.get_pretemp_scores(scores, choices, temperature.get_value())
            for i in range(num_nets):
                track_reward[epoch, i] = torch.mean(torch.sum(rewards[i], dim=1)).detach().cpu()
                track_reward_by_pos[epoch, :, i] = torch.mean(rewards[i], dim=0).detach().cpu()
                track_confidence[epoch, :, i] = torch.mean(pretemp_scores[i], dim=0).detach().cpu()
                if get_target_reward:
                    track_network_reward[epoch, :, i] = torch.sum(rewards[i], dim=1).detach().cpu()

        # save dataset-specific variables
        epoch_state = dict(
            epoch=epoch,
            batch=batch,
            scores=scores,
            choices=choices,
            loss=loss if get_loss else None,
            rewards=rewards if get_reward else None,
            gamma_transform=None,
            temperature=temperature.get_value(),
        )
        dataset.save_testing_variables(dataset_variables, epoch_state, **parameters)

        # update temperature (probably never used, but here anyway)
        temperature.step()

    # return training data
    results = dict(
        loss=track_loss if get_loss else None,
        reward=track_reward if get_reward else None,
        reward_by_pos=track_reward_by_pos if get_reward else None,
        confidence=track_confidence if get_reward else None,
        target_reward=track_target_reward if get_target_reward else None,
        network_reward=track_network_reward if get_target_reward else None,
        dataset_variables=dataset_variables,
    )

    return results
