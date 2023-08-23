from typing import Callable, Any
import importlib
from typing import Callable, Any
from pydoc import locate

creation_funcs = {}


def register(name: str, module):
    """Register modules such as datasets and models"""
    print("Registered")
    creation_funcs[name] = module


def load_module(name, config):
    module = importlib.import_module(config.module)  # type: ignore
    module.initialize()
    try:
        creation_func = creation_funcs[name]
        return creation_func(config)
    except KeyError:
        raise ValueError from None


class PluginInterface:
    """PluginInterface docstring"""

    @staticmethod
    def initialize() -> None:
        """Initialize the plugin"""


def import_module(name: str) -> PluginInterface:
    return importlib.import_module(name)  # type: ignore


def load_plugin(plugin_name: str) -> None:
    """Load the plugins"""
    plugin = import_module(plugin_name)
    plugin.initialize()
