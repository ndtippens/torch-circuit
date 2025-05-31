"""
torch-circuit: A PyTorch extension for building neural networks with skip connections and repeatable blocks.

This package provides the Circuit class and related components for creating
complex neural network architectures with named skip connections and repeatable blocks.
"""

from .circuit import Circuit, SaveInput, GetInput, StartBlock, EndBlock, LambdaLayer

__version__ = "0.9.1"
__author__ = "ntippens"
__email__ = "ndtippens@gmail.com"

__all__ = [
    "Circuit",
    "SaveInput", 
    "GetInput",
    "StartBlock",
    "EndBlock", 
    "LambdaLayer"
]
