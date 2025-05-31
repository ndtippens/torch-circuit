import torch
import torch.nn as nn
from typing import Dict, Any, List, Union, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict


class SaveInput(nn.Module):
    """Module to save intermediate activations for skip connections."""
    
    def __init__(self, name: str, transform: nn.Module | None = None):
        super().__init__()
        self.name = name
        self.transform = transform if transform is not None else nn.Identity()
    
    def forward(self, x):
        return x  # Pass through unchanged

class GetInput(nn.Module):
    """Module to get saved activations."""
    
    def __init__(self, name: str, op: Callable = torch.add, transform: nn.Module | None = None):
        super().__init__()
        self.name = name
        self.op = op
        self.transform = transform if transform is not None else nn.Identity()
    
    def forward(self, x):
        # instead of updating state here, we will do it in the forward pass of Circuit
        return x


class StartBlock(nn.Module):
    """Module to mark the start of a repeatable block."""
    
    def __init__(self, name: str, num_repeats: int = 1):
        super().__init__()
        self.name = name
        self.num_repeats = num_repeats
        
        if num_repeats < 1:
            raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")
    
    def forward(self, x):
        return x  # Pass through unchanged


class EndBlock(nn.Module):
    """Module to mark the end of a repeatable block."""
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    def forward(self, x):
        return x  # Pass through unchanged


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function
        
    def forward(self, x):
        return self.function(x)


class Circuit(nn.Module):
    """
    A neural network container that supports skip connections and repeatable blocks.
    
    Features:
    - SaveInput/GetInput: Named skip connections
    - StartBlock/EndBlock: Repeatable blocks to avoid manual layer duplication
    """
    
    def __init__(self, *layers):
        super().__init__()
        self.original_layers = layers
        self.expanded_layers = nn.ModuleList()
        self.skip_tensors = {}
        
        # Expand repeaters and build the actual layer sequence
        self._expand_repeaters()
    
    def _expand_repeaters(self):
        """Expand blocks into repeated layers."""
        i = 0
        repeat_stack = []  # Stack to handle nested repeats
        
        while i < len(self.original_layers):
            layer = self.original_layers[i]
            
            if isinstance(layer, StartBlock):
                # Find matching EndBlock
                end_idx = self._find_matching_end_block(i, layer.name)
                if end_idx == -1:
                    raise ValueError(f"No matching EndBlock found for StartBlock '{layer.name}'")
                
                # Get layers to repeat
                repeat_layers = list(self.original_layers[i+1:end_idx])
                
                # Add repeated blocks
                for repeat_idx in range(layer.num_repeats):
                    for repeat_layer in repeat_layers:
                        # Create unique names for skip connections in repeated blocks
                        if isinstance(repeat_layer, (SaveInput, GetInput)):
                            # Clone the skip layer with a unique name
                            unique_name = f"{repeat_layer.name}_repeat_{repeat_idx}"
                            if isinstance(repeat_layer, SaveInput):
                                new_layer = SaveInput(unique_name, repeat_layer.transform)
                            else: 
                                new_layer = GetInput(unique_name, repeat_layer.op, repeat_layer.transform)
                            self.expanded_layers.append(new_layer)
                        else:
                            # Regular layer - add as-is
                            self.expanded_layers.append(repeat_layer)
                
                # Skip to after EndBlock
                i = end_idx + 1
            
            elif isinstance(layer, EndBlock):
                # This should be handled by StartBlock
                raise ValueError(f"EndBlock '{layer.name}' without matching StartBlock")
            
            else:
                # Regular layer - add as-is
                self.expanded_layers.append(layer)
                i += 1
    
    def _find_matching_end_block(self, start_idx: int, name: str) -> int:
        """Find the matching EndBlock for a StartBlock."""
        nesting_level = 0
        
        for i in range(start_idx + 1, len(self.original_layers)):
            layer = self.original_layers[i]
            
            if isinstance(layer, StartBlock):
                nesting_level += 1
            elif isinstance(layer, EndBlock):
                if nesting_level == 0 and layer.name == name:
                    return i
                nesting_level -= 1
        
        return -1  # No matching EndBlock found
    
    def forward(self, x):
        """Forward pass through expanded layers with skip connection support."""
        self.skip_tensors = {}  # Reset skip tensors for this forward pass
        
        for layer in self.expanded_layers:
            if isinstance(layer, SaveInput):
                # Save the current activation (possibly transformed)
                saved_activation = layer.transform(x)
                self.skip_tensors[layer.name] = saved_activation
                # Continue with original activation
                
            elif isinstance(layer, GetInput):
                # Add the saved activation to current activation
                if layer.name not in self.skip_tensors:
                    raise RuntimeError(f"Skip connection '{layer.name}' not found. "
                                     f"Available: {list(self.skip_tensors.keys())}")
                input = self.skip_tensors[layer.name]
                x = layer.op( x, layer.transform(input) )
                
            else:
                # Regular layer - forward pass
                x = layer(x)
        
        return x
    
    def get_expanded_structure(self):
        """Get the expanded layer structure as a list of strings."""
        lines = []
        for i, layer in enumerate(self.expanded_layers):
            layer_name = layer.__class__.__name__
            if isinstance(layer, (SaveInput, GetInput)):
                layer_name += f"('{layer.name}')"
            lines.append(f"{i}: {layer_name}")
        return lines

        print("\nCircuit Structure:")
        last_block = None
        for idx, layer in enumerate(self.expanded_layers):
            # Detect block start/end for extra spacing
            if hasattr(layer, 'name') and hasattr(layer, 'num_repeats'):
                if last_block is not None:
                    print("")  # Blank line between blocks
                _print_layer(layer, idx, indent)
                last_block = layer.name
            elif hasattr(layer, 'name') and hasattr(layer, 'is_end') and layer.is_end:
                _print_layer(layer, idx, indent)
                print("")  # Blank line after block end
                last_block = None
            else:
                _print_layer(layer, idx, indent + (2 if last_block else 0))
        print("")  # Final blank line

    def visualize(self, figsize: Tuple[int, int] = (10, 12),
                  save_path: Optional[str] = None,
                  show_repeats: bool = True,
                  color_map: Optional[Dict[str, str]] = None):
        """
        Create a circuit diagram visualization of the network architecture.

        Args:
            figsize: Size of the figure (width, height)
            save_path: Path to save the figure. If None, the figure is displayed
            show_repeats: Whether to show repeat blocks as grouped components
            color_map: Dictionary mapping module types to colors
        """
        # Default color map for different module types
        default_colors = {
            'Linear': '#D0D0F0',  # Light blue
            'Conv': '#D0F0D0',    # Light green
            'Norm': '#F0F0D0',    # Light yellow
            'Attention': '#FFD580', # Light orange
            'SaveSkip': '#FFFACD', # Light yellow
            'AddSkip': '#FFFACD',  # Light yellow
            'Repeat': '#E6E6FA',   # Light purple
            'Pooling': '#D0F0F0',  # Light cyan
            'Dropout': '#F0D0F0',  # Light magenta
            'default': '#F8F8F8'   # Light gray
        }

        if color_map is None:
            color_map = default_colors
        else:
            merged = default_colors.copy()
            merged.update(color_map)
            color_map = merged

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Track skip connections
        skip_sources = {}
        skip_targets = {}

        # Analyze layers to determine structure
        layer_info = self._analyze_layers(show_repeats)
        n_layers = len(layer_info)

        # Improved vertical spacing and layout
        y_margin = 10
        y_start = 100 - y_margin
        y_end = y_margin
        total_height = y_start - y_end
        y_spacing = total_height / max(n_layers - 1, 1)

        x_center = 50
        box_width = 24
        box_height = 3

        # Track repeat block vertical lines
        repeat_lines = []
        layer_positions = []
        y_positions = []
        i = 0
        # Count only boxes (not Repeat/EndRepeat) for layer_positions
        box_indices = []
        while i < n_layers:
            layer = layer_info[i]
            y_pos = y_start - i * y_spacing
            y_positions.append(y_pos)
            if layer['type'] == 'Repeat':
                # Find contained layers
                block_name = layer['name']
                count = layer.get('count', 1)
                # Find start and end indices for this repeat block
                start_idx = i + 1
                end_idx = start_idx
                nest = 1
                while end_idx < n_layers:
                    if layer_info[end_idx]['type'] == 'Repeat' and layer_info[end_idx]['name'] == block_name:
                        nest += 1
                    elif layer_info[end_idx]['type'] == 'EndRepeat' and layer_info[end_idx]['name'] == block_name:
                        nest -= 1
                        if nest == 0:
                            break
                    end_idx += 1
                # Draw vertical line for repeat block
                y_top = y_start - i * y_spacing
                y_bot = y_start - end_idx * y_spacing
                x_line = x_center - 20
                ax.plot([x_line, x_line], [y_top, y_bot], color='purple', linewidth=2, zorder=1)

                # Add repeat count label
                ax.text(x_line - 2, (y_top + y_bot) / 2, f"{block_name} ×{count}", color='purple',
                        fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)

                # Skip drawing a box for the repeat block itself
                i += 1
                continue
            elif layer['type'] == 'EndRepeat':
                i += 1
                continue
            # Draw the layer box
            rect = patches.FancyBboxPatch(
                (x_center - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle=patches.BoxStyle("Round", pad=0.3),
                facecolor=self._get_layer_color(layer['type'], color_map), edgecolor='black', linewidth=1.2
            )
            ax.add_patch(rect)
            ax.text(x_center, y_pos, layer['name'], ha='center', va='center', fontsize=9, fontweight='medium')
            # Track skip connections
            if isinstance(layer['layer'], SaveInput):
                skip_sources[layer['layer'].name] = (x_center + box_width/2, y_pos, len(layer_positions))
            elif isinstance(layer['layer'], GetInput):
                skip_targets[layer['layer'].name] = (x_center - box_width/2, y_pos, len(layer_positions))
            layer_positions.append((x_center, y_pos, box_width, box_height))
            box_indices.append(i)
            i += 1

        # Draw skip connections as indirect, rectangular paths
        for name, (x2, y2, idx2) in skip_targets.items():
            if name in skip_sources:
                x1, y1, idx1 = skip_sources[name]
                # Route: right of source → outside right margin → down/up → left of target → target
                x_pad = 20
                x_right = x_center + box_width/2 + x_pad
                x_left = x_center - box_width/2 - x_pad
                path = [
                    (x1+1, y1),
                    (x_right, y1),
                    (x_right, y2),
                    (x1+1, y2)
                ]
                for j in range(len(path) - 1):
                    ax.plot([path[j][0], path[j+1][0]], [path[j][1], path[j+1][1]], color='black', linewidth=1.1, zorder=2)
                ax.annotate(
                    '', xy=path[-1], xytext=path[-2],
                    arrowprops=dict(arrowstyle="->", lw=1.1, color='black', shrinkA=0, shrinkB=0)
                )

                # Add centered and rotated skip connection label
                mid_x = (path[1][0] + path[2][0]) / 2  # Center x position
                mid_y = (path[1][1] + path[2][1]) / 2  # Center y position

                # Create a text box with rotation
                ax.text(mid_x+2, mid_y, name, fontsize=10, fontweight='bold',
                        rotation=270, ha='center', va='center')

        # Draw main flow arrows with spacing, outside and between layers
        for i in range(len(layer_positions) - 1):
            x0, y0, w0, h0 = layer_positions[i]
            x1, y1, w1, h1 = layer_positions[i+1]
            y0 -= 0.25
            y1 += 0.25
            ax.annotate(
                '',
                xy=(x1, y1 + h1/2),
                xytext=(x0, y0 - h0/2),
                arrowprops=dict(arrowstyle="->", lw=1.1, color='gray', shrinkA=0, shrinkB=0)
            )

        # Add title
        ax.text(x_center, 100 - y_margin/2, "Circuit Architecture", ha='center', va='bottom', fontsize=15, fontweight='bold')

        # Save or show the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def _analyze_layers(self, show_repeats: bool = True) -> List[Dict[str, Any]]:
        """Analyze layers to create a simplified representation for visualization."""
        layers = []
        if show_repeats:
            i = 0
            while i < len(self.original_layers):
                layer = self.original_layers[i]
                if isinstance(layer, StartBlock):
                    layers.append({
                        'type': 'Repeat',
                        'name': layer.name,
                        'layer': layer,
                        'count': layer.num_repeats
                    })
                    i += 1
                    continue
                elif isinstance(layer, EndBlock):
                    layers.append({
                        'type': 'EndRepeat',
                        'name': layer.name,
                        'layer': layer
                    })
                    i += 1
                    continue
                # Get a descriptive name for the layer
                layer_name = self._get_layer_name(layer)
                layer_type = self._get_layer_type(layer)
                layers.append({
                    'type': layer_type,
                    'name': layer_name,
                    'layer': layer
                })
                i += 1
        else:
            # Use expanded layers to show the full structure
            for layer in self.expanded_layers:
                layer_name = self._get_layer_name(layer)
                layer_type = self._get_layer_type(layer)

                layers.append({
                    'type': layer_type,
                    'name': layer_name,
                    'layer': layer
                })

        return layers

    def _get_layer_name(self, layer: nn.Module) -> str:
        """Get a descriptive name for a layer."""
        if isinstance(layer, SaveInput):
            return f"Save '{layer.name}'"
        elif isinstance(layer, GetInput):
            return f"Add '{layer.name}'"
        elif isinstance(layer, StartBlock):
            return f"Repeat '{layer.name}'"
        elif isinstance(layer, EndBlock):
            return f"End '{layer.name}'"
        else:
            # For standard PyTorch layers, use class name
            name = layer.__class__.__name__
            # Add shape info if available
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                name += f" ({layer.in_features}→{layer.out_features})"
            elif hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
                name += f" ({layer.in_channels}→{layer.out_channels})"
            return name

    def _get_layer_type(self, layer: nn.Module) -> str:
        """Get the type category of a layer for coloring."""
        if isinstance(layer, SaveInput):
            return 'SaveSkip'
        elif isinstance(layer, GetInput):
            return 'AddSkip'
        elif isinstance(layer, (StartBlock, EndBlock)):
            return 'Repeat'
        elif isinstance(layer, nn.Linear):
            return 'Linear'
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                                nn.ConvTranspose2d, nn.ConvTranspose3d)):
            return 'Conv'
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                                nn.InstanceNorm2d, nn.InstanceNorm3d,
                                nn.LocalResponseNorm)):
            return 'Norm'
        elif isinstance(layer, nn.MultiheadAttention):
            return 'Attention'
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
                                nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool1d,
                                nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                                nn.FractionalMaxPool2d, nn.FractionalMaxPool3d,
                                nn.LPPool1d, nn.LPPool2d)):
            return 'Pooling'
        elif isinstance(layer, (nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                                nn.AlphaDropout, nn.FeatureAlphaDropout)):
            return 'Dropout'
        else:
            return 'default'

    def _get_layer_color(self, layer_type: str, color_map: Dict[str, str]) -> str:
        """Get the color for a layer type."""
        for key in color_map:
            if key in layer_type:
                return color_map[key]
        return color_map['default']
