import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from natsort import natsorted
from freezedry import freezedry
from conditional_parser import ConditionalArgumentParser

import torch
import wandb
from matplotlib import pyplot as plt

from .. import files as files
from ..datasets import get_dataset
from ..utils import get_scheduler, scheduler_from_parser


class Experiment(ABC):
    def __init__(self, args=None) -> None:
        """Experiment constructor"""
        self.basename = self.get_basename()  # Register basename of experiment
        self.basepath = files.results_path() / self.basename  # Register basepath of experiment
        self.get_args(args=args)  # Parse arguments to python program
        self.register_timestamp()  # Register timestamp of experiment
        self.run = self.configure_wandb()  # Create a wandb run object (or None depending on args.use_wandb)
        self.device = self.args.device or "cuda" if torch.cuda.is_available() else "cpu"  # Register device for experiment

    def report(self, init=False, args=False, meta_args=False) -> None:
        """Method for programmatically reporting details about experiment"""
        # Report general details about experiment
        if init:
            print(f"Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {self.get_exp_path()}")
            print("using device: ", self.device)

            # Report any other relevant details
            if self.args.save_networks and self.args.nosave:
                print("Note: setting nosave to True will overwrite save_networks. Nothing will be saved.")

        # Report experiment parameters
        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args:
                    continue
                print(f"{key}={val}")

        # Report experiment meta parameters
        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args:
                    continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        """
        Method for registering formatted timestamp.

        If timestamp not provided, then the current time is formatted and used to identify this particular experiment.
        If the timestamp is provided, then that time is used and should identify a previously run and saved experiment.
        """
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.use_timestamp:
                self.args.timestamp = self.timestamp

    def get_dir(self, create=True) -> Path:
        """
        Method for return directory of target file using prepare_path.
        """
        # Make full path to experiment directory
        exp_path = self.basepath / self.get_exp_path()

        # Make experiment directory if it doesn't yet exist
        if create and not (exp_path.exists()):
            exp_path.mkdir(parents=True)

        return exp_path

    def get_exp_path(self) -> Path:
        """Method for returning child directories of this experiment"""
        # exp_path is the base path followed by whatever folders define this particular experiment
        # (usually things like ['network_name', 'dataset_name', 'test', 'etc'])
        exp_path = Path("/".join(self.prepare_path()))

        # if requested, will also use a timestamp to distinguish this run from others
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp

        return exp_path

    def get_path(self, name, create=True) -> Path:
        """Method for returning path to file"""
        # get experiment directory
        exp_path = self.get_dir(create=create)

        # return full path (including stem)
        return exp_path / name

    def configure_wandb(self):
        """create a wandb run file and set environment parameters appropriately"""
        if not self.args.use_wandb:
            return None

        wandb.login()
        run = wandb.init(
            project=self.get_basename(),
            name=f"Experiment: {self.timestamp}",
            config=self.args,
        )

        if str(self.basepath).startswith("/n/home"):
            # ATL Note 240223: We can update the "startswith" list to be
            # a registry of path locations that require WANDB_MODE to be offline
            # in a smarter way, but I think that using /n/ is sufficient in general
            os.environ["WANDB_MODE"] = "offline"

        return run

    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        See ``get_dir()`` for details.
        """
        pass

    def get_args(self, args=None):
        """
        Method for defining and parsing arguments.

        This method defines the standard arguments used for any Experiment, and
        the required method make_args() is used to add any additional arguments
        specific to each experiment.
        """
        self.meta_args = []  # a list of arguments that shouldn't be updated when loading an old experiment
        parser = ConditionalArgumentParser(description=f"arguments for {self.basename}")
        parser = self.make_args(parser)

        # saving and new experiment loading parameters
        parser.add_argument(
            "--nosave",
            default=False,
            action="store_true",
            help="prevents saving of results or plots",
        )
        parser.add_argument(
            "--justplot",
            default=False,
            action="store_true",
            help="plot saved data without retraining and analyzing networks",
        )
        parser.add_argument(
            "--save_networks",
            default=False,
            action="store_true",
            help="if --nosave wasn't provided, will also save networks that are trained",
        )
        parser.add_argument(
            "--showprms",
            default=False,
            action="store_true",
            help="show parameters of previously saved experiment without doing anything else",
        )
        parser.add_argument(
            "--showall",
            default=False,
            action="store_true",
            help="if true, will show all plots at once rather than having the user close each one for the next",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="which device to use (automatic if not provided)",
        )

        # add meta arguments
        self.meta_args += ["nosave", "justplot", "save_networks", "showprms", "showall", "device"]

        # common parameters that shouldn't be updated when loading old experiment
        parser.add_argument(
            "--use_timestamp",
            default=False,
            action="store_true",
            help="if used, will save data in a folder named after the current time (or whatever is provided in --timestamp)",
        )
        parser.add_argument(
            "--timestamp",
            default=None,
            help="the timestamp of a previous experiment to plot or observe parameters",
        )

        # parse arguments (passing directly because initial parser will remove the "--experiment" argument)
        self.args = parser.parse_args(args=args)

        # manage device
        if self.args.device is None:
            self.args.device = "cuda" if torch.cuda.is_available() else "cpu"

        # do checks
        if self.args.use_timestamp and self.args.justplot:
            assert self.args.timestamp is not None, "if use_timestamp=True and plotting stored results, must provide a timestamp"

    @abstractmethod
    def make_args(self, parser) -> ConditionalArgumentParser:
        """
        Required method for defining special-case arguments.

        This should just use the add_argument method on the parser provided as input.
        """
        pass

    def get_prms_path(self):
        """Method for loading path to experiment parameters file"""
        return self.get_dir() / "prms.pth"

    def get_results_path(self):
        """Method for loading path to experiment results files"""
        return self.get_dir() / "results.pth"

    def get_network_path(self, name):
        """Method for loading path to saved network file"""
        network_path = self.get_dir() / "networks"
        if not network_path.exists():
            network_path.mkdir()
        return network_path / f"{name}.pt"

    def get_checkpoint_path(self):
        """Method for loading path to network checkpoint file directory"""
        return self.get_dir()

    def _update_args(self, prms, verbose=False):
        """Method for updating arguments from saved parameter dictionary"""
        # First check if saved parameters contain unknown keys
        if prms.keys() > vars(self.args).keys():
            raise ValueError(f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(self.args).keys())}")

        # Then update self.args while ignoring any meta arguments
        for ak in vars(self.args):
            if ak in self.meta_args:
                continue  # don't update meta arguments
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                if verbose:
                    print(f"Requested argument {ak}={vars(self.args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(self.args, ak, prms[ak])

    def save_repo(self, ignore_docs=True, verbose=False):
        """Method for saving a copy of the code repo at the time this experiment was run"""
        local_repo_path = files.local_repo_path()
        target_path = self.get_dir() / "frozen_repo.zip"
        regexp_ignore = ["docs/*"] if ignore_docs else []
        freezedry(local_repo_path, target_path, ignore_git=True, use_gitignore=True, regexp_ignore=regexp_ignore, verbose=verbose)

    def save_experiment(self, results):
        """Method for saving experiment parameters and results to file"""
        # Save experiment parameters
        torch.save(vars(self.args), self.get_prms_path())
        # Save experiment results
        torch.save(results, self.get_results_path())

    def load_experiment(self, use_saved_prms=True, no_results=False, verbose=False):
        """Method for loading saved experiment parameters and results"""
        # Check if prms path is there
        if not self.get_prms_path().exists():
            raise ValueError(f"saved parameters at: f{self.get_prms_path()} not found!")

        # Check if results directory is there
        if not self.get_results_path().exists():
            raise ValueError(f"saved results at: f{self.get_results_path()} not found!")

        # Load parameters into object
        if use_saved_prms:
            prms = torch.load(self.get_prms_path())
            self._update_args(prms, verbose=verbose)

        # Don't load results if requested
        if no_results:
            return None

        # Load and return results
        return torch.load(self.get_results_path())

    def save_networks(self, nets, id=None):
        """
        Method for saving any networks that were trained

        Names networks with index in list of **nets**
        If **id** is provided, will use id in addition to the index
        """
        name = f"net_{id}_" if id is not None else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net.state_dict(), self.get_network_path(cname))

    def load_networks(self, nets, id=None, check_number=True):
        """
        Method for loading any networks that were trained

        This only works by loading the state_dict, so we have to provided instantiated networks first.
        It assumes that number of saved networks correspond to the loaded nets (in a natsort kind of way),
        so the check_number=True argument makes sure the number of requested networks (len(nets))= the
        number of detected saved networks.

        If **id** is provided, will use id in addition to the index to name the network.
        """
        name = f"net_{id}_" if id is not None else "net_"
        pattern = self.get_network_path(name + "*").name
        matches = natsorted([match.stem for match in self.get_dir().rglob(pattern)])
        if check_number:
            msg = f"the number of detected networks with name signature {name}*.pt does not match the number of requested networks ({len(matches)}/{len(nets)})"
            assert len(matches) == len(nets), msg
        for idx, match in enumerate(matches):
            c_state_dict = torch.load(self.get_network_path(match))
            nets[idx].load_state_dict(c_state_dict)
        return nets

    @abstractmethod
    def main(self) -> Tuple[Dict, List[torch.nn.Module]]:
        """
        Required method for operating main experiment functions.

        This method should perform any core training and analyses related to the experiment
        and return a results dictionary and a list of pytorch nn.Modules. The second requirement
        (torch modules) can probably be relaxed, but doesn't need to yet so let's keep it as is
        for overall clarity.
        """
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Required method for operating main plotting functions.

        Should accept as input a results dictionary and run plotting functions.
        If any plots are to be saved, then each plotting function must do so
        accordingly.
        """
        pass

    # -- support for main processing loop --
    def prepare_dataset(self):
        """simple method for getting dataset"""
        # return the dataset
        dataset_parameters = vars(self.args).copy()
        dataset_parameters.pop("task", None)
        dataset_parameters.pop("device", None)
        return get_dataset(self.args.task, build=True, device=self.device, **dataset_parameters)

    def make_train_parameters(self, dataset, train=True, **parameter_updates):
        """simple method for getting training parameters"""
        # get the training/testing parameters
        parameters = {}
        parameters["train"] = train  # controls dataset batch generation
        parameters["num_epochs"] = self.args.train_epochs if train else self.args.test_epochs
        parameters["device"] = self.device
        parameters["verbose"] = not getattr(self.args, "silent", False)
        if train:
            parameters["learning_mode"] = getattr(self.args, "learning_mode")
        parameters["max_possible_output"] = dataset.get_max_possible_output()
        parameters["reward_gamma"] = getattr(self.args, "reward_gamma", 1.0)

        # handle network state parameters -- temperature should be scheduled for training and 1.0 always for testing
        parameters["thompson"] = getattr(self.args, "thompson", True) if train else False
        if train:
            parameters["temperature"] = scheduler_from_parser(self.args, "train_temperature")
        else:
            parameters["temperature"] = get_scheduler("constant", build=True, initial_value=1.0)

        # Saving parameters
        parameters["save_loss"] = getattr(self.args, "save_loss", False)
        parameters["save_reward"] = getattr(self.args, "save_reward", False)

        # Handle return_target
        # (If using supervised learning or saving loss, then return_target=True, otherwise False)
        # (If you want to test target reward, use parameter_updates to request this in the caller function)
        parameters["return_target"] = getattr(self.args, "learning_mode", "") == "supervised" or getattr(self.args, "save_loss", False)

        # Handle baseline parameters
        parameters["baseline"] = getattr(self.args, "baseline", False) if train else False
        if parameters["baseline"]:
            parameters["bl_temperature"] = self.args.bl_temperature
            parameters["bl_thompson"] = self.args.bl_thompson
            parameters["bl_significance"] = self.args.bl_significance
            parameters["bl_batch_size"] = self.args.bl_batch_size
            parameters["bl_frequency"] = self.args.bl_frequency

        # Handle checkpointing parameters (only used for training)
        if train:
            parameters["use_prev_ckpts"] = getattr(self.args, "use_prev_ckpts", False)
            parameters["save_ckpts"] = getattr(self.args, "save_ckpts", False) and not self.args.nosave
            if parameters["save_ckpts"]:
                parameters["uniq_ckpts"] = self.args.uniq_ckpts
                parameters["freq_ckpts"] = self.args.freq_ckpts
                parameters["path_ckpts"] = self.get_checkpoint_path()

        # Handle wandb parameters
        parameters["use_wandb"] = getattr(self.args, "use_wandb")

        # additionally update parameters directly if requested
        parameters.update(parameter_updates)

        # return train loop parameters
        return parameters

    def plot_ready(self, name):
        """standard method for saving and showing plot when it's ready"""
        # if saving, then save the plot
        if not self.args.nosave:
            plt.savefig(str(self.get_path(name)))
        if self.run is not None:
            self.run.log({name: wandb.Image(plt)})
        # show the plot now if not doing showall
        if not self.args.showall:
            plt.show()
