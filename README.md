# torch-circuit

A PyTorch extension for building neural networks with skip connections and repeatable blocks.

## Features

- **Named Skip Connections**: Easily implement named ResNet-style skip connections
- **Repeatable Blocks**: Simple, repeatable block definitions
- **Transformations**: Apply abitrary operations and transformations to skip connections
- **PyTorch Compatible**: Works seamlessly with existing PyTorch code and training loops
- **Visualization**: Generate circuit diagrams of your network architecture

## Installation

### From Source

```bash
git clone https://github.com/ndtippens/torch-circuit.git
cd torch-circuit
pip install -e .
```

### From PyPI

```bash
pip install torch-circuit
```

## Quick Start

```python
import torch
import torch.nn as nn
from torch_circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock

# Create a simple ResNet-style model with skip connections
model = Circuit(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
        
    # Repeatable ResNet block
    StartBlock("resnet", num_repeats=3),
    SaveInput("residual", transformation=None),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    GetInput("residual", op=torch.add),
    EndBlock("resnet"),
    
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# Produces standard PyTorch model
x = torch.randn(1, 3, 32, 32)
output = model(x)
# example output:
# tensor([[ 1.2898,  0.7074, -0.2531, -0.5240, -2.1423,  0.8159,  0.7738,  1.0178,
#         -0.5311,  1.2696]], grad_fn=<AddmmBackward0>)
```
See examples/resnet_mnist.py for details and a full performance comparison.

## Key Components

### Circuit

The main container class that supports skip connections and repeatable blocks.

### SaveInput / GetInput

- `SaveInput(name, transformation=None)`: Save the input tensor with a given name
- `GetInput(name, op=torch.add, transformation=None)`: Retrieve saved tensor and combine it with the current tensor using an operation (e.g. addition, concatenation). Optionally, you may also transform the retrieved tensor before combining it with the current tensor.

### StartBlock / EndBlock

- `StartBlock(name, num_repeats=N)`: Mark the beginning of a repeatable block
- `EndBlock(name)`: Mark the end of a repeatable block

## Examples

See the `examples/` directory for a complete example demonstrating equivalence to standard PyTorch implementations:

- `examples/resnet_mnist.py`: ResNet architecture on MNIST dataset

## Architecture Visualization

torch-circuit can generate simple visual diagrams of your network architecture:

```python
model.visualize(save_path="example.png")
```
![Example Circuit](https://github.com/ndtippens/torch-circuit/blob/main/examples/example.png)

## Advanced Usage

### Custom Operations

You can use arbitrary operations for combining skip connections:

```python
# Concatenate residual signals in a specified dimension
GetInput("residual", op=lambda x, y: torch.cat((x, y), dim=1))

# Element-wise multiplication
GetInput("residual", op=torch.mul)
```
