# standard imports
import torch

# dominoes package
from ..networks import get_pointer_network, get_pointer_methods, get_pointer_arguments
from .base import Experiment
from . import arglib
from ..utils import named_transpose
from ..plotting import plot_train_test_results


class PointerParameterComparison(Experiment):
    """An experiment class for running pointer architecture parameter comparisons"""

    def get_basename(self):
        """basename for experiment (determines folder, filenames, etc.)"""
        return "ptr_prm_comp"

    def prepare_path(self):
        """prepare the path for the experiment"""
        return [self.args.task, self.args.learning_mode]

    def make_args(self, parser):
        """Method for adding experiment specific arguments to the argument parser"""
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_network_training_metaparameters(parser)
        parser = arglib.add_scheduling_parameters(parser, "lr")
        parser = arglib.add_scheduling_parameters(parser, "train_temperature")
        parser = arglib.add_pointernet_parameters(parser)
        parser = arglib.add_pointer_layer_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dataset_parameters(parser)
        return parser

    def create_networks(self, input_dim, context_parameters):
        """
        method for creating networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """

        # create networks
        embedding_dim, pointer_kwargs = get_pointer_arguments(self.args)

        raise NotImplementedError("Need to code up some variation in parameters for this experiment.")

        nets = [get_pointer_network(input_dim, embedding_dim, **context_parameters, **pointer_kwargs) for _ in range(self.args.replicates)]
        nets = [net.to(self.device) for net in nets]
        optimizers = [torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.wd) for net in nets]

        # The only thing that needs to be recorded is the pointer method used for each network
        prms = dict()
        return nets, optimizers, prms

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """
        # load dataset
        dataset = self.prepare_dataset()

        # input dimensionality
        input_dim = dataset.get_input_dim()
        context_parameters = dataset.get_context_parameters()

        # create networks
        nets, optimizers, prms = self.create_networks(input_dim, context_parameters)

        # train and test networks
        results = dataset.train_and_test(self, nets, optimizers, target_reward=True, do_training=True, do_testing=True)

        # add parameters to results
        results["prms"] = prms

        # return results and trained networks
        return results, nets

    def plot(self, results):
        """
        main plotting loop
        """
        pointer_methods = self.pointer_methods()
        plot_train_test_results(self, results["train_results"], results["test_results"], pointer_methods)
