from abc import ABC, abstractmethod


def _check_scheduler(scheduler_name):
    """check if schedulers is in the registry"""
    if scheduler_name not in SCHEDULER_REGISTRY:
        valid_schedulers = list(SCHEDULER_REGISTRY.keys())
        raise ValueError(f"Scheduler ({scheduler_name}) is not in SCHEDULER_REGISTRY, valid schedulers are: {valid_schedulers}")


def get_scheduler(scheduler_name, build=False, initial_value=None, **kwargs):
    """
    lookup scheduler_name constructor from scheduler_name registry by name

    builds scheduler_name and returns a scheduler_name object using any kwargs
    otherwise just returns the class constructor
    """
    scheduler_name = scheduler_name.lower()
    _check_scheduler(scheduler_name)
    scheduler = SCHEDULER_REGISTRY[scheduler_name]
    if build:
        # initial value is required for all schedulers
        assert initial_value is not None, "initial_value must be provided to build a scheduler"

        # each scheduler has additional requirements
        requirements = SCHEDULER_REQUIREMENTS[scheduler_name]

        # build additional requirements into an args list
        args = []
        for requirement in requirements:
            if requirement in kwargs and kwargs[requirement] is not None:
                args.append(kwargs.pop(requirement))
            else:
                raise ValueError(f"Scheduler ({scheduler_name}) requires argument ({requirement})")

        # return constructed scheduler
        return scheduler(initial_value, *args, **kwargs)

    # return scheduler constructor
    return scheduler


def scheduler_from_parser(args, name, initial_value=None, negative_clip=True):
    """
    get scheduler arguments from parser

    in ptrseq.experiments.arglib.add_scheduling_parameters, the arguments are added to the parser
    with a prefix "name" that determines which value to make a scheduler for. This function will
    look for those arguments and retrieving them by using name as a prefix.

    It will then build the scheduler using the retrieved arguments and return the scheduler object.

    When the initial value isn't expected to be provided by the user, it can be passed as a keyword
    argument (but the initial_value from the args will take precedence).

    args:
        args: argparse.Namespace object
        name: str, name of the scheduler
    """
    # convert args to dictionary
    args_dict = vars(args)

    # check if the scheduler parameters are included in the parser
    if f"{name}_scheduler" not in args_dict:
        # if not, return a constant scheduler with the initial value
        initial_value = initial_value or args_dict[name]
        return get_scheduler("constant", build=True, initial_value=initial_value, negative_clip=negative_clip)

    # if included, get the name, and check if it is a valid scheduler
    scheduler_name = args_dict[f"{name}_scheduler"].lower()
    _check_scheduler(scheduler_name)

    # get universal arguments for the scheduler
    initial_value = args_dict[f"{name}_initial_value"] or initial_value or args_dict[name]
    negative_clip = args_dict[f"{name}_negative_clip"] or negative_clip

    # get required arguments for the scheduler
    requirements = SCHEDULER_REQUIREMENTS[scheduler_name]
    extra_args = {key: args_dict[f"{name}_{key}"] for key in requirements}

    # return constructed scheduler
    return get_scheduler(scheduler_name, build=True, initial_value=initial_value, negative_clip=negative_clip, **extra_args)


class _Scheduler(ABC):
    """
    Base class for all schedulers

    Increments epochs and an internal value that can be retrieved by the user.
    Requires child classes to implement the _step_value method for updating the value
    in different ways depending on the kind of scheduling.
    """

    def __init__(self, initial_value, negative_clip=True):
        """initialize the scheduler"""
        self.initial_value = initial_value
        self.current_value = initial_value
        self.current_epoch = 0
        self.negative_clip = negative_clip

    def reset(self):
        """reset the scheduler to its initial value and epoch"""
        self.current_value = self.initial_value
        self.current_epoch = 0

    def get_value(self):
        """get the current value of the scheduler"""
        return self.current_value

    def set_value(self, value):
        """set the current value of the scheduler"""
        self.current_value = value

    def get_epoch(self):
        """get the current epoch of the scheduler"""
        return self.current_epoch

    def set_epoch(self, epoch, with_value_update=True):
        """set the current epoch of the scheduler"""
        if with_value_update:
            self.step(epoch=epoch)
        else:
            self.current_epoch = epoch

    @abstractmethod
    def _step_value(self):
        """required method for updating the value of the scheduler"""
        raise NotImplementedError

    def step(self, epoch=None):
        """
        method for updating the value of the scheduler

        if epoch is none, will increment the current epoch by 1,
        otherwise will set the value to the given epoch
        """
        # update epoch
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # update value
        next_value = self._step_value()
        if self.negative_clip and self.current_epoch < 0:
            next_value = self.initial_value

        # update value
        self.current_value = next_value


class StepScheduler(_Scheduler):
    """
    -- written to be similar to pytorch! some documentation copied directly: --

    Decays the value of the stored parameter by gamma every step_size epochs.
    Explicitly references to epoch 0, such that if the initial value is 0.1 and
    the step_size is 30, if you set the epoch to -10 and then call step(), the
    new value will be 1.0.

    Example:
    ```python
    lr_scheduler = StepScheduler(initial_value=0.1, step_size=30, gamma=0.1)
    lr = 0.1     if epoch < 30
    lr = 0.01    if 30 <= epoch < 60
    lr = 0.001   if 60 <= epoch < 90
    for epoch in range(100):
        train(..., lr=lr_scheduler.get_value())
        lr_scheduler.step()
    """

    def __init__(self, initial_value, step_size, gamma, negative_clip=True):
        super().__init__(initial_value, negative_clip=negative_clip)
        self.step_size = step_size
        self.gamma = gamma

    def _step_value(self):
        """update the value of the scheduler"""
        return self.initial_value * self.gamma ** (self.current_epoch // self.step_size)


class ExponentialScheduler(_Scheduler):
    """
    -- written to be similar to pytorch! some documentation copied directly: --

    Decays the value of the stored parameter by gamma every epoch.

    Explicitly references to epoch 0, such that if the initial value is 0.1,
    if you set the epoch to -2 and then call step(), the new value will be 1.0.
    (Because the epoch will be adjusted to -1 before stepping).

    Example:
    ```python
    lr_scheduler = ExponentialScheduler(initial_value=0.1, gamma=0.01)
    for epoch in range(100):
        train(..., lr=lr_scheduler.get_value())
        lr_scheduler.step()
    """

    def __init__(self, initial_value, gamma, negative_clip=True):
        super().__init__(initial_value, negative_clip=negative_clip)
        self.gamma = gamma

    def _step_value(self):
        """update the value of the scheduler"""
        return self.initial_value * self.gamma**self.current_epoch


class ExponentialBaselineScheduler(_Scheduler):
    """
    -- written to be similar to pytorch! some documentation copied directly: --

    Decays the value of the stored parameter by gamma every epoch towards a shifted baseline.

    Explicitly references to epoch 0, such that if the initial value is 0.1,
    if you set the epoch to -2 and then call step(), the new value will be 1.0.
    (Because the epoch will be adjusted to -1 before stepping).

    Example:
    ```python
    lr_scheduler = ExponentialBaselineScheduler(initial_value=2.0, final_value=1.0, gamma=0.01)
    for epoch in range(100):
        train(..., lr=lr_scheduler.get_value())
        lr_scheduler.step()
    """

    def __init__(self, initial_value, final_value, gamma, negative_clip=True):
        super().__init__(initial_value, negative_clip=negative_clip)
        self.final_value = final_value
        self.value_range = self.initial_value - self.final_value
        self.gamma = gamma

    def _step_value(self):
        """update the value of the scheduler"""
        return self.final_value + self.value_range * self.gamma**self.current_epoch


class LinearScheduler(_Scheduler):
    """
    -- written to be similar to pytorch! some documentation copied directly: --

    Decays the value of the stored parameter on a linear trajectory until reaching the final value.
    Clipped between initial and final, such that negative epochs will return the initial value and
    epochs beyond the total_epochs will return the final value.

    Example:
    ```python
    lr_scheduler = LinearScheduler(initial_value=0.1, gamma=0.01)
    for epoch in range(100):
        train(..., lr=lr_scheduler.get_value())
        lr_scheduler.step()
    """

    def __init__(self, initial_value, final_value, total_epochs, negative_clip=True):
        super().__init__(initial_value, negative_clip=negative_clip)
        self.final_value = final_value
        self.total_epochs = total_epochs
        self.value_range = self.initial_value - self.final_value

    def _get_value_fraction(self):
        """get the fraction of the way through the schedule"""
        return 1 - min(max(self.current_epoch / self.total_epochs, 0.0), 1.0)

    def _step_value(self):
        """update the value of the scheduler"""
        return self.final_value + self.value_range * self._get_value_fraction()


class ConstantScheduler(_Scheduler):
    """
    -- written to be similar to pytorch! some documentation copied directly: --

    Returns the same value every epoch. For consistent code, this is included as a scheduler.

    Example:
    ```python
    lr_scheduler = ConstantScheduler(initial_value=0.1)
    for epoch in range(100):
        train(..., lr=lr_scheduler.get_value())
        lr_scheduler.step()
    """

    def __init__(self, initial_value, negative_clip=True):
        super().__init__(initial_value, negative_clip=negative_clip)

    def _step_value(self):
        """update the value of the scheduler"""
        return self.initial_value


SCHEDULER_REGISTRY = {
    "step": StepScheduler,
    "exp": ExponentialScheduler,
    "expbase": ExponentialBaselineScheduler,
    "linear": LinearScheduler,
    "constant": ConstantScheduler,
}

SCHEDULER_REQUIREMENTS = {
    "step": ["step_size", "gamma"],
    "exp": ["gamma"],
    "expbase": ["final_value", "gamma"],
    "linear": ["final_value", "total_epochs"],
    "constant": [],
}

if SCHEDULER_REGISTRY.keys() != SCHEDULER_REQUIREMENTS.keys():
    raise ValueError("SCHEDULER_REGISTRY and SCHEDULER_REQUIREMENTS must have the same keys")
