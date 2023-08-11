import os, sys
from pydoc import locate


def load_module(config):
    module = locate(config.module)
    if not module:
        raise AttributeError()
    return module(config)
