"""
torch-circuit: A PyTorch extension for building neural networks with skip connections and repeatable blocks.

This package provides the Circuit class and related components for creating
complex neural network architectures with named skip connections and repeatable blocks.
"""

from .circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock, LambdaLayer

__version__ = "0.1.0"
__author__ = "torch-circuit contributors"
__email__ = "your-email@example.com"

__all__ = [
    "Circuit",
    "SaveInput", 
    "GetInput",
    "StartBlock",
    "EndBlock", 
    "LambdaLayer"
]
