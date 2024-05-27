import sys
from copy import deepcopy
from inspect import signature
from argparse import ArgumentParser


class ConditionalArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        """
        Initialize the ConditionalArgumentParser object.

        args and kwargs are passed directly to the ArgumentParser Object. See
        ArgumentParser documentation for more information.
        """
        super(ConditionalArgumentParser, self).__init__(*args, **kwargs)
        self._conditional_parents = []
        self._conditional_func_match = []
        self._conditional_args = []
        self._conditional_kwargs = []
        self._num_conditional = 0

    def parse_args(self, args=None, namespace=None):
        """Parse the arguments and return the namespace."""
        if args is None:
            args = sys.argv[1:]

        _parser = deepcopy(self)
        already_added = [False for _ in range(self._num_conditional)]
        _parser = self._add_conditionals(_parser, args, already_added)

        return ArgumentParser.parse_args(_parser, args=args, namespace=namespace)

    def add_conditional(self, dest, func_match, *args, **kwargs):
        """
        add conditional argument that is only added when parent arguments match a condition

        args:
            dest: is the destination of the parent (where to look in the namespace)
            func_match: is a function that value of the destination and returns a boolean
                        indicating whether or not to add this conditional
                        note: if it callable, then it will be called on the value of dest
                              if it isn't callable, then it will simply by compared to the value of dest
            *args: the arguments to add when the condition is met
            **kwargs: the keyword arguments to add when the condition is met
        """
        # attempt to add the conditional argument to a dummy parser to check for errors
        _dummy = deepcopy(self)
        _dummy.add_argument(*args, **kwargs)

        # if it passes, store the details to the conditional argument
        self._conditional_parents.append(dest)
        self._conditional_func_match.append(self._make_callable(func_match))
        self._conditional_args.append(args)
        self._conditional_kwargs.append(kwargs)
        self._num_conditional += 1

    def _add_conditionals(self, _parser, args, already_added):
        """Add conditional arguments to the parser through a hierarchical parse."""
        # remove help arguments for an initial parse to determine if conditionals are needed
        args = [arg for arg in args if arg not in ["-h", "--help"]]
        namespace = ArgumentParser.parse_known_args(_parser, args=args)[0]

        # whenever conditionals aren't ready, add whatever is needed then try again
        if not self._conditionals_ready(namespace, already_added):
            # for each conditional, check if it is required and add it if it is
            for i, parent in enumerate(self._conditional_parents):
                if self._conditional_required(namespace, parent, already_added, i):
                    # add conditional argument
                    _parser.add_argument(*self._conditional_args[i], **self._conditional_kwargs[i])
                    already_added[i] = True

            # recursively call the function until all conditionals are added
            _parser = self._add_conditionals(_parser, args, already_added)

        # return a parser with all conditionals added
        return _parser

    def _make_callable(self, func):
        """make a function that returns a boolean from a function or value."""
        # if the function provided is callable, use it as is
        if callable(func):
            if len(signature(func).parameters.values()) == 2:
                return func
            else:
                return lambda dest_value, namespace: func(dest_value)

        # otherwise, create a function that compares the value to the provided value
        else:
            return lambda dest_value, namespace: dest_value == func

    def _conditionals_ready(self, namespace, already_added):
        """Check if all conditionals are finished."""
        # for each conditional, if it is required and not already added, return False
        for idx, parent in enumerate(self._conditional_parents):
            if self._conditional_required(namespace, parent, already_added, idx):
                return False

        # if all required conditionals are added, return True
        return True

    def _conditional_required(self, namespace, parent, already_added, idx):
        """check if a conditional is required to be added"""
        # first check if the parent exists in the namespace
        if hasattr(namespace, parent):
            # then check if this conditional has already been added
            if not already_added[idx]:
                # if it hasn't been added and the conditional function matches the value in parent,
                # then return True to indicate that this conditional is required
                if self._conditional_func_match[idx](getattr(namespace, parent), namespace):
                    return True

        # otherwise return False to indicate that this conditional does not need to be added
        return False


# copy the docstring and signature from ArgumentParser for more useful help messages
ConditionalArgumentParser.__init__.__doc__ = ArgumentParser.__init__.__doc__
ConditionalArgumentParser.__init__.__signature__ = signature(ArgumentParser.__init__)


def check_args(method_name, args, required_args):
    """method for checking args for required arguments and returning useful errors (args is a dictionary)"""
    for key in required_args:
        if key not in args:
            raise ValueError(f"required arg {key} not found in args ({method_name} requires {required_args})")


def process_arguments(args, required_args, required_kwargs, possible_kwargs, name):
    """method for getting the required and optional kwargs from stored argument dictionary"""
    # if any required args are missing, raise an error
    check_args(name, args, required_args)

    # get required args (in order of list!)
    rq_args = [args[arg] for arg in required_args]

    # get kwargs
    kwargs = {}

    # if any required kwargs are missing, raise an error
    check_args(name, args, required_kwargs)
    for key, value in required_kwargs.items():
        kwargs[value] = args[key]

    # if any kwargs are included in args, add them to the dictionary
    for key, value in possible_kwargs.items():
        if key in args:
            kwargs[value] = args[key]

    return rq_args, kwargs


def build_args(*kargs, kvargs={}, append_hyphens=True):
    """method for building a valid set of arguments from a dictionary"""
    assert isinstance(kvargs, dict), "kvargs must be a dictionary"
    arg_list = []
    for key in kargs:
        arg_list.append(key)
    for key, value in kvargs.items():
        arg_list.append(f"--{key}" if append_hyphens else key)
        arg_list.append(value)
    return arg_list
