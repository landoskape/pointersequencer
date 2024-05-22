import torch


def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))


def named_transpose(list_of_lists):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)
    """
    return map(list, zip(*list_of_lists))


def compute_stats_by_type(tensor, num_types, dim, method="var"):
    """
    helper method for returning the mean and variance across a certain dimension
    where multiple types are concatenated on that dimension

    for example, suppose we trained 2 networks each with 3 sets of parameters
    and concatenated the loss in a tensor like [set1-loss-net1, set1-loss-net2, set2-loss-net1, ...]
    then this would contract across the nets from each set and return the mean and variance
    """
    num_on_dim = tensor.size(dim)
    num_per_type = int(num_on_dim / num_types)
    tensor_by_type = tensor.unsqueeze(dim)
    expand_shape = list(tensor_by_type.shape)
    expand_shape[dim + 1] = num_per_type
    expand_shape[dim] = num_types
    tensor_by_type = tensor_by_type.view(expand_shape)
    type_means = torch.mean(tensor_by_type, dim=dim + 1)
    if method == "var":
        type_dev = torch.var(tensor_by_type, dim=dim + 1)
    elif method == "std":
        type_dev = torch.std(tensor_by_type, dim=dim + 1)
    elif method == "se":
        type_dev = torch.std(tensor_by_type, dim=dim + 1) / torch.sqrt(num_per_type)
    else:
        raise ValueError(f"Method ({method}) not recognized.")

    return type_means, type_dev


def argsort(seq):
    """sort a list by value, return index to sort order"""
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def check_similarity(d1, d2, name1="dict1", name2=None, compare_shapes=False, compare_dims=False):
    # check if results and ckpt results have same structure
    name2 = name2 or name1[:-1] + "2"
    assert d1.keys() == d2.keys(), f"{name1} and {name2} do not have matching keys"
    for key in d1:
        if isinstance(d1[key], torch.tensor):
            assert isinstance(d2[key], torch.tensor), f"{name1}[{key}] and {name2}[{key}] do not match in their type"
            if compare_dims:
                assert d1[key].dim() == d2[key].dim(), f"{name1}[{key}] and {name2}[{key}] do not match in their dim"
            if compare_shapes:
                assert d1[key].shape == d2[key].shape, f"{name1}[{key}] and {name2}[{key}] do not match in their shape"
        elif isinstance(d1[key], dict):
            check_similarity(d1[key], d2[key])
        else:
            assert type(d1[key]) == type(d2[key]), f"d1[{key}] and d2[{key}] do not match in their type"
            print(f"found other type in {name1}: {key}, type={type(d1[key])}. Not checking for similarity with checkpoint.")
