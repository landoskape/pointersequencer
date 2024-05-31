# Imports
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import torch
from rastermap import Rastermap

from ..experiments import get_experiment
from ..utils import build_args
from ..networks.net_utils import forward_batch
from ..files import local_repo_path


class EncodingRepresentationAnalysis:
    def __init__(self, args=None):
        args = self.get_args(args)
        self.task = args.task
        self.save = args.save
        self.noshow = args.noshow

    def main(self):
        """main analysis loop"""
        cache, choices, prms = self.get_data(self.task)
        for idx in range(len(prms["pointer_methods"])):
            self.analyze_encodings(cache, choices, prms, idx)

    def get_args(self, args):
        parser = ArgumentParser(description="Analyze encoding representations")
        parser.add_argument("--task", type=str, default="dominoe_sorter", help="Task to analyze")
        parser.add_argument("--save", default=False, action="store_true")
        parser.add_argument("--noshow", default=False, action="store_true")
        return parser.parse_args(args=args)

    def get_fig_path(self, name):
        base_path = local_repo_path() / "docs" / "media" / "encoding_representations"
        fig_path = base_path / (name + ".png")
        if not fig_path.parent.exists():
            fig_path.parent.mkdir(parents=True)
        return fig_path

    def figure_ready(self, fig_name):
        if self.save:
            fig_path = self.get_fig_path(fig_name)
            plt.savefig(fig_path)

        if not self.noshow:
            plt.show()

    def get_data(self, task="dominoe_sorter"):
        torch.set_grad_enabled(False)

        experiment = "ptr_arch_comp"
        args = dict(task=task, encoder_method="attention", decoder_method="attention", replicates="3", embedding_bias="False")
        exp = get_experiment(experiment, build=True, args=build_args(kvargs=args))

        _ = exp.load_experiment(use_saved_prms=False, verbose=True)
        dataset = exp.prepare_dataset()

        # input dimensionality
        input_dim = dataset.get_input_dim()
        context_parameters = dataset.get_context_parameters()

        # create networks
        nets, _, prms = exp.create_networks(input_dim, context_parameters)
        nets = exp.load_networks(nets)
        for net in nets:
            net.eval()

        # prepare dataset parameters
        parameters = exp.make_train_parameters(dataset, train=False)
        # dominoes = dataset.get_dominoe_set(parameters["train"])

        # create example batch
        batch = dataset.generate_batch(**(parameters | {"batch_size": 1024}))

        # run data through the network
        max_possible_output = parameters.get("max_possible_output")  # this is the maximum number of outputs ever
        _, choices, cache = forward_batch(nets, batch, max_possible_output, temperature=1.0, thompson=False, cache=True)

        # measure rewards
        # rewards = [dataset.reward_function(choice, batch) for choice in choices]

        return cache, choices, prms

    def analyze_encodings(self, cache, choices, prms, idx):
        name = prms["pointer_methods"][idx]

        # sort encoded data by choice order
        c_encoded = cache[idx]["hook_encoder"]
        c_choice = choices[idx]
        s_encoded = torch.gather(c_encoded, 1, c_choice.unsqueeze(2).expand(-1, -1, c_encoded.size(2)))
        sz_encoded = (s_encoded - s_encoded.mean((0, 1), keepdim=True)) / s_encoded.std((0, 1), keepdim=True)
        sz_encoded = sz_encoded.detach().cpu().numpy()
        s_choice = torch.sort(c_choice, dim=1).values

        # reshape z-scored data into all_tokens x embedding_dim
        at_encoded = sz_encoded.reshape(-1, sz_encoded.shape[2])
        at_choice = s_choice.view(-1).detach().cpu().numpy()

        ichoice = np.argsort(at_choice)

        # run rastermap
        prms = dict(
            n_PCs=64,
            n_clusters=12,
            locality=0.75,
            time_lag_window=5,
        )

        # fit rastermap on tokens
        model = Rastermap(**prms).fit(at_encoded)
        y = model.embedding  # neurons x 1
        isort = model.isort

        # fit rastermap on embedding dimension
        model = Rastermap(**prms).fit(at_encoded.T)
        jsort = model.isort

        # show encoded data average
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
        ax[0].imshow(sz_encoded.mean(0).T[jsort], interpolation=None, aspect="auto")
        ax[0].set_xlabel("Token Order (Sorted By Network Choice)")
        ax[0].set_ylabel("Embedding Dimension (Sorted By Rastermap)")
        ax[0].set_title(f"Encoded Data Average ({name}:{idx})")
        ax[1].plot(range(sz_encoded.shape[1]), sz_encoded.mean(0)[:, jsort])
        ax[1].set_xlabel("Token Order (Sorted By Network Choice)")
        ax[1].set_ylabel("Embedding Activation (z-scored)")
        ax[1].set_title(f"Average Activation ({name}:{idx})")

        self.figure_ready(f"encoded_data_average_{name}_{idx}")

        # choose sort method
        sort_tokens_by_choice = True
        itoken = ichoice if sort_tokens_by_choice else isort

        # plot
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), width_ratios=(5, 1), layout="constrained")
        ax[0].imshow(at_encoded[itoken][:, jsort], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
        ax[0].set_xlabel("Embedding Dimension (Sorted By Rastermap)")
        ax[0].set_ylabel("Token Order (Sorted By Network Choice)")
        ax[0].set_title(f"Encoded Data ({name}:{idx})")
        ax[1].plot(at_choice[itoken], np.arange(len(at_choice)), color="black")
        ax[1].set_xlabel("Token Order (Sorted By Network Choice)")
        ax[1].set_ylabel("Token Index")
        ax[1].set_title("Network Choice")

        self.figure_ready(f"encoded_data_all_tokens_{name}_{idx}")
