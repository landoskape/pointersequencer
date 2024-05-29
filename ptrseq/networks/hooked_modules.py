from contextlib import contextmanager
import torch


class HookedModule(torch.nn.Module):
    def __init__(self):
        super(HookedModule, self).__init__()
        self.cache = {}
        self._store_cache = False
        self._layer_to_name = {}

    def _register_hook(self, module, name=None):
        def _forward_hook(module, input, output):
            if self.store_hidden:
                name = self._layer_to_name[module] = name
                self.cache[name] = output

        module.register_forward_hook(hook)

    def _add_hooks(self):
        self._layer_to_name = {}
        for name, layer in self.model.named_children():
            self._layer_to_name[layer] = name
            if True:  # not isinstance(layer, nn.Sequential):
                layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        """Hook to store hidden layer output in cache with a given name"""
        if self.store_hidden:
            self.cache[self._layer_to_name[module]] = output

    @contextmanager
    def _handle_cache(self, store_hidden):
        self.store_hidden = store_hidden
        try:
            yield
        finally:
            self.store_hidden = False

    def forward(self, x, store_hidden=False):
        with self._handle_cache(store_hidden):
            output = self.model(x)
        return output
