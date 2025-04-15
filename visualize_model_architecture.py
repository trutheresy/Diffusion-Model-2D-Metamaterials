import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


def visualize_model_architecture(model: nn.Module, input_size: Optional[Tuple[int, ...]] = None, device: str = 'cpu') -> None:
    """
    Visualizes the architecture of a PyTorch model by printing information about each layer.
    
    Args:
        model: The PyTorch model to visualize
        input_size: Optional input size to the model (excluding batch dimension). If None, will attempt to infer from model.
        device: The device to run the model on ('cpu' or 'cuda')
    """
    # Try to determine input size if not provided
    if input_size is None:
        # Check if model has an attribute that might indicate input size
        if hasattr(model, 'input_size'):
            input_size = model.input_size
        elif hasattr(model, 'input_shape'):
            input_size = model.input_shape
        elif hasattr(model, 'input_dim'):
            # Handle case where input_dim might be a single integer
            if isinstance(model.input_dim, int):
                input_size = (model.input_dim,)
            else:
                input_size = model.input_dim
        else:
            # Default to a reasonable size if we can't determine
            # This is a fallback and might not work for all models
            input_size = (3, 224, 224)  # Common image input size (channels, height, width)
            print(f"Warning: Could not determine input size, using default: {input_size}")
    
    # Create a dummy input tensor
    x = torch.rand(1, *input_size).to(device)
    model = model.to(device)
    model.eval()
    
    # Dictionary to store outputs of each layer
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    
    # Register hooks for all modules
    hooks = []
    for name, layer in model.named_modules():
        if name and not any(child_name.startswith(name + '.') for child_name, _ in model.named_modules()):
            hooks.append(layer.register_forward_hook(get_activation(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print model architecture
    print("\n" + "="*80)
    print(f"MODEL ARCHITECTURE VISUALIZATION")
    print("="*80)
    
    total_params = 0
    
    for name, layer in model.named_modules():
        if name and not any(child_name.startswith(name + '.') for child_name, _ in model.named_modules()):
            # Get layer output shape
            output_shape = tuple(activation[name].shape)
            
            # Get number of parameters
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            total_params += params
            
            # Print layer information
            print(f"\nLAYER: {name}")
            print(f"TYPE: {layer.__class__.__name__}")
            print(f"PARAMETERS: {params:,}")
            print(f"OUTPUT SHAPE: {output_shape}")
            print("-"*50)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print(f"TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    print("="*80 + "\n")
