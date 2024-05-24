import matplotlib.pyplot as plt
import matplotlib as mpl


def _get_x(idx, xOffset=[-0.2, 0.2]):
    return [xOffset[0] + idx, xOffset[1] + idx]


def train_test_plot(
    train_mean,
    train_se,
    test_mean,
    test_se,
    labels,
    name,
    figdim=5,
    figratio=2,
    alpha=0.3,
    cmap="tab10",
    ylim=None,
    curriculum_epochs=None,
):
    num_types = len(labels)
    colormap = mpl.colormaps[cmap]
    width_ratios = [figdim, figdim / figratio]

    fig, ax = plt.subplots(1, 2, figsize=(sum(width_ratios), figdim), width_ratios=width_ratios, layout="constrained", sharey=True)
    for idx, label in enumerate(labels):
        ax[0].plot(range(train_mean.size(0)), train_mean[:, idx], color=colormap(idx), label=label)
        ax[0].fill_between(
            range(train_mean.size(0)),
            train_mean[:, idx] + train_se[:, idx],
            train_mean[:, idx] - train_se[:, idx],
            color=(colormap(idx), alpha),
        )
        ax[1].plot(_get_x(idx), [test_mean[idx]] * 2, color=colormap(idx), label=label, lw=4)
        ax[1].plot([idx, idx], [test_mean[idx] - test_se[idx], test_mean[idx] + test_se[idx]], color=colormap(idx), lw=1.5)

    if curriculum_epochs is not None:
        for epoch in curriculum_epochs:
            ax[0].axvline(epoch, color="black", linestyle="--", lw=1)

    ax[0].set_xlabel("Training Epoch")
    ax[0].set_ylabel(name)
    ax[0].set_title("Training " + name)
    ax[0].set_ylim(ylim)
    ylims = ax[0].get_ylim()

    ax[1].set_xticks(range(num_types))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax[1].set_title("Testing " + name)
    ax[1].set_xlim(-0.5, num_types - 0.5)
    ax[1].set_ylim(ylims)

    return fig, ax
