# PyTorch ML Template

This is a first attempt to create a standardized ML template for running 
ML experiments. 

Problem it aims to solve: 
- Often we need to run 1 model with multiple configurations 
- or 1 model against mulitiple datasets 
- Dynamically import modules when needed, instead of
    1 gigantic import at the top.

Doing this in the main training loop is a pain and makes things 
very hard to unit test, deal with and generically makes it very hard 
to read / modify. 

So this repo is build on a plugin-architecture. 
That is, everything *can* be imported via a plugin or during testing
via the regular import statements to make things flexible. 
Moreover, once you have a config file, you load the config file and load 
all modules / datasets dynamically. 
The latter means that the main loop does not need to know what model,
dataset etc you are using, so it becomes very clean without and without 
the need to add any implementation details in the main loop. 

# Usage
An example to get started is provided in the following set of files: 
- `main.py`
- dataset is implemented in `datasets.custom_dataset`
- model is implemented in `models.custom_model`

For the model, implement it how you would normally, but add all the init inputs 
in a separate config dataclass. 
The config dataclass has to be for the form 

```python
@dataclass 
MyConfig:
"""
The module variable is the module path 
to where the module is implemented. 
""" 
    module: str = "folder.file" 
    arg_1: str = "myarg"
    arg_2: int = 1
    etc ...
```

Please find below a simple example:
Files are found in:

- `main.py`
- `datasets/custom_dataset.py`
- `models/custom_model.py`

If you would like to run a full experiment, 
run `main.py` and check out the code therein.

## Usage in the main file

```python
from datasets.custom_dataset import MnistConfig
from models.custom_model import NNConfig
from loaders.factory import load_module
"""
Normally one would write the configs to a file 
and load them. See generate_experiments.py for 
an example.
"""

model = load_module("model", NNConfig())
dataset = load_module("dataset", MnistConfig())

for (X, y) in dataset.test_dataloader():
    print(model(X.squeeze(1)))
    break
```

## Model definition

```python

import torch.nn.functional as F
from torch import nn

from models.base_model import BaseModel
from loaders.factory import register
from dataclasses import dataclass


@dataclass
class NNConfig:
    module: str = "models.custom_model"
    input_size: int = 28*28
    hidden1_size: int = 250
    hidden2_size: int = 250
    output_size: int = 10


class NN(BaseModel):
    def __init__(self,config: NNConfig):
        super(NN, self).__init__(config)
        self.linear1 = nn.Linear(config.input_size, config.hidden1_size)
        self.linear2 = nn.Linear(config.hidden1_size, config.hidden2_size)
        self.linear3 = nn.Linear(config.hidden2_size, config.output_size)

    def forward(self, X):
        X = X.view(-1,28*28)
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return F.log_softmax(X, dim=1)


def initialize():
    register("model",NN)

```

## Dataset definition

```python
from datasets.base_dataset import DataModule
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split

from dataclasses import dataclass
from loaders.factory import register


@dataclass
class MnistConfig:
    module: str = "datasets.custom_dataset"
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 0


class MnistDataModule(DataModule):
    """
    Example implementation for an existing dataset.
    Note that we do the transform here, that is why we
    need to create a separate class for the "new"
    dataset.
    """
    def __init__(self,config: MnistConfig):
        super().__init__(config.data_dir,config.batch_size,config.num_workers)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        entire_dataset = MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )


def initialize():
    register("dataset",MnistDataModule)
```


## Warning
This is still the first iteration so not it is a working but not polished 
repository. The dataset base class needs some polishing.
Feel free to add propose changes! Happy to discuss!

## TODO
- Structure of the main.py is reasonable but still needs to made to work with above example. 
- Need to add standard unit tests. 
- Polish dataset base class e.g. remove setup() method. 
- Add the experiments in the form of Errica https://arxiv.org/abs/1912.09893
- Make it in the form of a git template 


