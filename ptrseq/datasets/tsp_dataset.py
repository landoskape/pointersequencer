from copy import copy
import torch


from .base import Dataset, DatasetSL, DatasetRL, RequiredParameter
from .support import make_path
from ..utils import process_arguments


class TSPDataset(Dataset, DatasetSL, DatasetRL):
    """A dataset for generating traveling salesman problem environments for training and evaluation"""

    task = "tsp"

    def __init__(self, device="cpu", **parameters):
        """constructor method"""
        # first add loss function setup to the supervised loss component of this class
        super().__init__()

        self.task = "tsp"

        self.set_device(device)

        # set parameters to required defaults first, then update
        self.prms = self.get_default_parameters()
        init_prms = self.process_arguments(parameters)  # get initialization parameters
        self.prms = self.parameters(**init_prms)  # update reference parameters for the dataset

    def get_default_parameters(self):
        """
        return the class parameters for the task. This is hard-coded here and only here,
        so if the parameters change, this method should be updated.

        None means the parameter is required and doesn't have a default value. Otherwise,
        the value is the default value for the parameter.

        returns:
            list of str, the required parameters for the task
        """
        # tsp parameters
        params = dict(
            num_cities=RequiredParameter(),  # this parameter is required to be set at initialization
            coord_dims=2,
            batch_size=1,
            return_target=False,
            ignore_index=-100,
            threads=1,
        )
        return params

    def process_arguments(self, args):
        """process arguments (e.g. from an ArgumentParser) for this dataset and set to parameters"""
        required_args = []
        required_kwargs = dict(
            num_cities="num_cities",
        )
        possible_kwargs = dict(
            coord_dims="coord_dims",
            batch_size="batch_size",
            return_target="return_target",
            ignore_index="ignore_index",
            threads="threads",
        )
        init_prms = process_arguments(args, required_args, required_kwargs, possible_kwargs, self.__class__.__name__)[1]
        return init_prms

    def get_input_dim(self, coord_dims=None):
        """
        get the input dimension of the dataset

        args (optional, uses default registered at initialization if not provided):
            coord_dims: int, the number of dimensions for the coordinates

        returns:
            int, the input dimension of the dataset
        """
        input_dim = coord_dims or self.prms["coord_dims"]
        return input_dim

    def get_context_parameters(self):
        """
        get the parameters of the contextual/multimodal inputs for the dataset

        returns:
            dict, the type of the context input for the dataset (see Pointer constructor)
        """
        context_parameters = dict(
            contextual=False,
            multimodal=False,
            num_multimodal=0,
            mm_input_dim=None,
            require_init=True,
            permutation=True,
        )
        return context_parameters

    def create_training_variables(self, num_nets, **train_parameters):
        """dataset specific training variable storage"""
        return {}  # nothing here yet, but ready for it in the future

    def save_training_variables(self, training_variables, epoch_state, **train_parameters):
        """dataset specific training variable storage"""
        pass  # nothing to do (usually update training_variables in place)

    def create_testing_variables(self, num_nets, **test_parameters):
        """dataset specific testing variable storage"""
        return {}  # nothing here yet, but ready for it in the future

    def save_testing_variables(self, testing_variables, epoch_state, **test_parameters):
        """dataset specific testing variable storage"""
        pass  # nothing to do (update testing_variables in place)

    def get_max_possible_output(self):
        """
        get the maximum possible output for the dataset

        returns:
            int, the maximum possible output for the dataset (it's just the number of cities minus 1 for the start city)
        """
        return self.prms["num_cities"] - 1

    @torch.no_grad()
    def generate_batch(self, device=None, **kwargs):
        """
        generates a batch of TSP environments with the specified parameters and additional outputs

        parallelized preparation of batch, better to use 1 thread if num_cities~<10 or batch_size<=256

        batch keys:
            input: torch.Tensor, the input to the network, as a binary dominoe representation (and null token)
            train: bool, whether the batch is for training or evaluation
            selection: torch.Tensor, the selection of dominoes in the hand
            target: torch.Tensor, the target for the network (only if requested)
        """
        # get device
        device = self.get_device(device)

        # get parameters with potential updates
        prms = self.parameters(**kwargs)

        input = torch.rand((prms["batch_size"], prms["num_cities"], prms["coord_dims"]), dtype=torch.float)
        dists = torch.cdist(input, input)

        # define initial position as closest to origin (arbitrary but standard choice)
        init = torch.argmin(torch.sum(input**2, dim=2), dim=1)

        # construct batch dictionary
        input = self.input_to_device(input, device=device)
        dists = self.input_to_device(dists, device=device)
        init = self.input_to_device(init, device=device)
        batch = dict(input=input, dists=dists, init=init)

        # add task specific parameters to the batch dictionary
        batch.update(prms)

        if prms["return_target"]:
            batch["target"] = self.input_to_device(self.get_paths(input, dists, init, prms["threads"]), device=device)

        return batch

    def get_paths(self, coordinates, distances, init, threads=1):
        """
        for batch of (batch, num_cities, 2), returns shortest path using
        held-karp algorithm that ends closest to origin and is clockwise
        """
        if threads > 1:
            if not hasattr(self, "persistent_pool"):
                self.create_persistent_pool(threads)

            # get paths
            path = list(self.persistent_pool_starmap(make_path, zip(coordinates, distances, init)))

        else:
            path = [make_path(coord, dist, idx) for coord, dist, idx in zip(coordinates, distances, init)]

        return torch.stack(path).long()

    def target_as_choice(self, target, ignore_index=None):
        """
        convert the target to a choice based on the ignore index

        args:
            target: torch.Tensor, the target to convert to a choice
            ignore_index: int, the index to ignore in the target (default is None, will use dataset.prms["ignore_index"])

        returns:
            torch.Tensor, the choice based on the target
        """
        ignore_index = ignore_index or self.prms["ignore_index"]
        if torch.any(target == ignore_index):
            raise ValueError("ignore_index should not be in the target for the traveling salesman problem!")
        return target

    @torch.no_grad()
    def reward_function(self, choices, batch, new_city_scalar=0.0, **kwargs):
        """
        measure the reward acquired by the choices made by a set of networks for the current batch

        rewards have two components, a distance component and a new city component

        1. the distance component is the negative of the distance traveled between cities for each step
        including the implicit return to the initial city at the end of the sequence (added to the total
        distance traveled in the last step).

        2. the new city component is 1 for new cities and -1 for visited cities. This is used to penalize
        the network from returning to a city that has already been visited. In practice, pointer networks
        usually use a permutation rule (with prior city masking) to ensure that the network doesn't return
        to any previous city, so the new city component is multiplied by a scalar (default 0) to make it
        irrelevant to the reward.

        args:
            choice: torch.Tensor, index to the choices made by the network
            batch: tuple, the batch of data generated for this training step
            new_city_scalar: float, the scalar to multiply the new city component by
            kwargs: not used, here for consistency with other dataset types

        returns:
            torch.Tensor, the rewards for the network (batch_size, choices)
        """
        assert choices.ndim == 2, "choices should be a 2-d tensor of the sequence of choices for each batch element"
        num_cities = batch["num_cities"]
        batch_size, num_choices = choices.shape
        device = choices.device

        distance = torch.zeros((batch_size, num_choices)).to(device)
        new_city = torch.ones((batch_size, num_choices)).to(device)

        last_location = batch["init"]  # last (i.e. initial position) is preset initial position
        src = torch.ones((batch_size, 1), dtype=torch.bool).to(device)  # src tensor for updating other tensors
        visited = torch.zeros((batch_size, num_cities), dtype=torch.bool).to(device)  # boolean for whether city has been visited
        visited.scatter_(1, last_location.view(batch_size, 1), src)  # put first city in to the "visited" tensor
        for nc in range(num_choices):
            # get index to next city
            next_location = choices[:, nc]
            # this is a batched vector of possible distances from previous city to all other cities
            c_dist_possible = torch.gather(batch["dists"], 1, last_location.view(batch_size, 1, 1).expand(-1, -1, num_cities)).squeeze(1)
            # this is a vector (over the batch) of the distance between previous and next city
            distance[:, nc] = torch.gather(c_dist_possible, 1, next_location.view(batch_size, 1)).squeeze(1)
            # get whether the next city has been visited
            c_visited = torch.gather(visited, 1, next_location.view(batch_size, 1)).squeeze(1)
            # update visited tensor with next city
            visited.scatter_(1, next_location.view(batch_size, 1), src)
            # new_city is a tensor of 1s and -1s, where 1s are new cities and -1s are visited cities
            new_city[c_visited, nc] = -1.0
            new_city[~c_visited, nc] = 1.0
            # update last location for next choices
            last_location = copy(next_location)

        # the traveling salesman defines an initial city, and assumes that the agent travels through all
        # other cities in a permutation then returns to the initial city... so that final step needs to
        # be represented somewhere in the reward function.
        # here, we just add the implicit distance traveled to that final choice
        c_dist_possible = torch.gather(batch["dists"], 1, batch["init"].view(batch_size, 1, 1).expand(-1, -1, num_cities)).squeeze(1)
        distance[:, -1] += torch.gather(c_dist_possible, 1, choices[:, -1].view(batch_size, 1)).squeeze(1)

        # combine two reward types
        reward = -distance + new_city_scalar * new_city

        return reward
