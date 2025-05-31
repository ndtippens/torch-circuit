# torch-circuit

A PyTorch extension for building neural networks with skip connections and repeatable blocks.

## Features

- **Named Skip Connections**: Easily implement ResNet-style skip connections with named references
- **Repeatable Blocks**: Define blocks once and repeat them multiple times without code duplication
- **Visualization**: Generate circuit diagrams of your network architecture
- **PyTorch Compatible**: Works seamlessly with existing PyTorch code and training loops

## Installation

### From Source

```bash
git clone https://github.com/your-username/torch-circuit.git
cd torch-circuit
pip install -e .
```

### From PyPI (when published)

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
    SaveInput("residual"),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    GetInput("residual", op=torch.add),  # Add skip connection
    nn.ReLU(),
    EndBlock("resnet"),
    
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# Use like any PyTorch model
x = torch.randn(1, 3, 32, 32)
output = model(x)

# Visualize the architecture
model.visualize(save_path="resnet_example.pdf")
```

## Key Components

### Circuit

The main container class that supports skip connections and repeatable blocks.

### SaveInput / GetInput

- `SaveInput(name)`: Save the input tensor with a given name
- `GetInput(name, op=torch.add)`: Retrieve saved tensor and combine it with the current tensor using an operation (e.g. addition, concatenation)

### StartBlock / EndBlock

- `StartBlock(name, num_repeats=N)`: Mark the beginning of a repeatable block
- `EndBlock(name)`: Mark the end of a repeatable block

## Examples

See the `examples/` directory for a complete example demonstrating equivalence to standard PyTorch implementations:

- `examples/resnet_mnist.py`: ResNet architecture on MNIST dataset

## Architecture Visualization

torch-circuit can generate simple visual diagrams of your network architecture:

```python
model.visualize(save_path="architecture.pdf")
```

## Advanced Usage

### Custom Operations

You can use custom operations for combining skip connections:

```python
# Element-wise multiplication instead of addition
GetInput("residual", op=torch.mul)

# Custom lambda function
GetInput("residual", op=lambda x, y: x + 0.5 * y)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

