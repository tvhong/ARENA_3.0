# %%

import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import torchinfo
except:
    %pip install torchinfo jaxtyping

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")

# %%

import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line

# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.maximum(t.tensor(0), x)


tests.test_relu(ReLU)

# %%
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.weight = nn.Parameter(self.rand(in_features, (out_features, in_features)))

        if bias:
            self.bias = nn.Parameter(self.rand(in_features, (out_features, )))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        return t.matmul(x, self.weight.T) +  \
                (self.bias if self.bias is not None else t.tensor(0.0))

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}"

    def rand(self, in_features, shape):
        bound = 1.0 / math.sqrt(in_features)
        return bound * (2 * t.rand(*shape) - 1)

tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as the size of the flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
    
# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28*28, 100, bias=True)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10, bias=True)
        # self.layers = nn.ModuleList([
        #     Flatten(0, -1),
        #     Linear(28*28, 100, bias=True),
        #     ReLU(),
        #     Linear(100, 10, bias=True)
        # ])
        # for layer in self.layers:
        #     print(layer)
        #     print(list(layer.parameters()))

    def forward(self, x: Tensor) -> Tensor:
        y = x
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        # for layer in self.layers:
        #     y = layer(y)

        return y


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%

MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "chapter0_fundamentals/exercises/data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()
# %%

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# If this is CPU, we recommend figuring out how to get cuda access (or MPS if you're on a Mac).
print(device)

# %%
model = SimpleMLP().to(device)

batch_size = 128
epochs = 3

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

# for epoch in range(epochs):
#     pbar = tqdm(mnist_trainloader)

#     for imgs, labels in pbar:
#         # Move data to device, perform forward pass
#         imgs, labels = imgs.to(device), labels.to(device)
#         logits = model(imgs)

#         # Calculate loss, perform backward pass
#         loss = F.cross_entropy(logits, labels)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         # Update logs & progress bar
#         loss_list.append(loss.item())
#         pbar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{loss:.3f}")

# line(
#     loss_list,
#     x_max=epochs * len(mnist_trainset),
#     labels={"x": "Examples seen", "y": "Cross entropy loss"},
#     title="SimpleMLP training on MNIST",
#     width=700,
# )
# %%
@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g. self.batch_size=64.
    Any of these fields can also be overridden when you create an instance, e.g. SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3

def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], list[float], SimpleMLP]:
    """
    Trains the model, using training parameters from the `args` object. Returns the model, and lists of loss & accuracy.
    """
    # YOUR CODE HERE - add a validation loop to the train function from above
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_list = []
    accuracy_list = []
    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch+1}/{args.epochs}", loss=f"{loss:.3f}")

        # Validation loop
        accurate_preds = 0
        for imgs, labels in mnist_testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)
            accurate_preds += (logits.argmax(dim=-1) == labels).sum().item()

        accuracy_list.append(accurate_preds / len(mnist_testset))

    return loss_list, accuracy_list, model


# args = SimpleMLPTrainingArgs()
# loss_list, accuracy_list, model = train(args)

# line(
#     y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
#     use_secondary_yaxis=True,
#     x_max=args.epochs * len(mnist_trainset),
#     labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
#     title="SimpleMLP training on MNIST",
#     width=800,
# )
# %%

class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # YOUR CODE HERE - define & initialize `self.weight`
        self.weight = nn.Parameter(self.rand((out_channels, in_channels, kernel_size, kernel_size)))


    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])
    
    def rand(self, size) -> Tensor:
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_size * self.kernel_size)
        # l + (r - l) * t.rand(*size)
        # = -bound + (bound - (-bound)) * t.rand(*size)
        # = -bound + 2 * bound * t.rand(*size)
        # = bound(2 * t.rand(*size) - 1)
        return bound * (2 * t.rand(*size) - 1)


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")
# %%

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])
# %%

class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x
# %%

class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """
        if self.training:
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True) 
            var = t.var(x, unbiased=False, dim=(0, 2, 3), keepdim=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * einops.rearrange(mean, '1 c 1 1 -> c')
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * einops.rearrange(var, '1 c 1 1 -> c')
            self.num_batches_tracked += 1
        else:
            mean = einops.rearrange(self.running_mean, 'c -> 1 c 1 1')
            var = einops.rearrange(self.running_var, 'c -> 1 c 1 1')

        x_norm = (x - mean) / t.sqrt(var + self.eps) 

        w = einops.rearrange(self.weight, "w -> 1 w 1 1")
        b = einops.rearrange(self.bias, "b -> 1 b 1 1")
        x_affine = x_norm * w + b

        return x_affine

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}"


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)
