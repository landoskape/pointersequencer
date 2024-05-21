import matplotlib.pyplot as plt
import matplotlib as mpl

from .utils import compute_stats_by_type, train_test_plot


def plot_train_results(exp, results, labels, name="train"):
    """
    Simplest plot method for plotting loss/reward (or something else eventually) across epochs

    Assumes that the loss/rewards are divided into len(labels) types across the first dimension
    (see below for variable names), will make a plot of the mean across epochs for each type and
    label it accordingly.

    The experiment object passed in as the first argument determines if the plot is saved or shown.
    """
    num_types = len(labels)

    if "loss" in results and results["loss"] is not None:
        loss = compute_stats_by_type(results["loss"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(loss[:, i], label=labels[i])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{name} loss")
        plt.legend(fontsize=8)

        exp.plot_ready(f"{name}_loss")

    if "reward" in results and results["reward"] is not None:
        reward = compute_stats_by_type(results["reward"], num_types, 1)[0]
        for i in range(num_types):
            plt.plot(reward[:, i], label=labels[i])
        plt.xlabel("Epochs")
        plt.ylabel("Reward")
        plt.title(f"{name} reward")
        plt.legend(fontsize=8)

        exp.plot_ready(f"{name}_reward")


def plot_train_test_results(exp, train_results, test_results, labels):
    """
    Plot reward/loss for training and testing results. Plot reward comparison with target.

    Assumes that the loss/rewards are divided into len(labels) types across the first dimension
    (see below for variable names), will make a plot of the mean across epochs for each type and
    label it accordingly.

    The experiment object passed in as the first argument determines if the plot is saved or shown.
    """
    num_types = len(labels)
    plot_loss = "loss" in train_results and "loss" in test_results and train_results["loss"] is not None and test_results["loss"] is not None
    plot_reward = (
        "reward" in train_results and "reward" in test_results and train_results["reward"] is not None and test_results["reward"] is not None
    )

    if plot_loss:
        train_mean, train_se = compute_stats_by_type(train_results["loss"], num_types, 1)
        test_mean, test_se = compute_stats_by_type(test_results["loss"].mean(dim=0), num_types, 0)

        fig, ax = train_test_plot(train_mean, train_se, test_mean, test_se, labels, "loss", ylim=(0, None))

        exp.plot_ready("train_test_loss")

    if plot_reward:
        train_mean, train_se = compute_stats_by_type(train_results["reward"], num_types, 1)
        test_mean, test_se = compute_stats_by_type(test_results["reward"].mean(dim=0), num_types, 0)

        fig, ax = train_test_plot(train_mean, train_se, test_mean, test_se, labels, "reward")

        exp.plot_ready("train_test_reward")
