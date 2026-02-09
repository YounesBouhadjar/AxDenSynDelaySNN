# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Standard Spiking Neural Network model.

This code is modified from:
https://github.com/Thvnvtos/SNN-delays

Refactored SNN using layer-based structure similar to SiLIFLayer.
Follows the pattern from SSM-inspired-LIF repo.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from train import Model
from utils import set_seed
from models.layers import LIFLayer


class SNN(Model):
    """
    Multi-layered Spiking Neural Network using layer-based structure.
    Similar to SSM-inspired-LIF SNN class but adapted for spikingjelly framework.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    
    def build_model(self):
        """Build the model - renamed from _init_layers for compatibility with Model base class."""
        self.snn = self._init_layers()
    
    def _init_layers(self):
        """
        Initialize layers following SSM-inspired-LIF pattern.
        Returns nn.ModuleList of layer instances.
        """
        snn = nn.ModuleList([])
        input_size = self.config.n_inputs
        
        # Hidden layers
        for i in range(self.config.n_hidden_layers):
            snn.append(
                LIFLayer(
                    input_size=input_size,
                    hidden_size=self.config.n_hidden_neurons,
                    config=self.config,
                    is_output_layer=False
                )
            )
            input_size = self.config.n_hidden_neurons
        
        # Output layer
        snn.append(
            LIFLayer(
                input_size=input_size,
                hidden_size=self.config.n_outputs,
                config=self.config,
                is_output_layer=True
            )
        )
        
        return snn
    
    def init_model(self):
        """Initialize model weights."""
        set_seed(self.config.seed)
        self.mask = []
        
        if self.config.init_w_method == 'kaiming_uniform':
            for i, layer in enumerate(self.snn):
                if hasattr(layer, 'W'):
                    torch.nn.init.kaiming_uniform_(layer.W.weight, nonlinearity='relu')
                    if self.config.sparsity_p > 0:
                        with torch.no_grad():
                            mask = torch.rand(layer.W.weight.size()).to(layer.W.weight.device)
                            mask[mask > self.config.sparsity_p] = 1
                            mask[mask <= self.config.sparsity_p] = 0
                            self.mask.append(mask)
                            layer.W.weight *= mask
        
        # Collect parameters for optimization (needed for Model base class)
        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []
        
        for layer in self.snn:
            # Linear weights
            if hasattr(layer, 'W'):
                self.weights.append(layer.W.weight)
                if self.config.bias:
                    self.weights_bn.append(layer.W.bias)
            
            # BatchNorm weights
            if hasattr(layer, 'norm') and layer.normalize:
                self.weights_bn.append(layer.norm.weight)
                self.weights_bn.append(layer.norm.bias)
            
            # Parametric LIF weights
            if hasattr(layer, 'neuron'):
                if isinstance(layer.neuron, nn.Module):
                    for name, param in layer.neuron.named_parameters():
                        if 'w' in name.lower() or 'tau' in name.lower():
                            self.weights_plif.append(param)
                    # Handle SiLIFNode parameters
                    if hasattr(layer.neuron, 'alpha_param') and layer.neuron.alpha_param is not None:
                        self.weights_plif.append(layer.neuron.alpha_param)
                    if hasattr(layer.neuron, 'beta_param') and layer.neuron.beta_param is not None:
                        self.weights_plif.append(layer.neuron.beta_param)
                    if hasattr(layer.neuron, 'dt') and layer.neuron.dt is not None:
                        self.weights_plif.append(layer.neuron.dt)
                    if hasattr(layer.neuron, 'a_param') and layer.neuron.a_param is not None:
                        self.weights_plif.append(layer.neuron.a_param)
                    if hasattr(layer.neuron, 'b_param') and layer.neuron.b_param is not None:
                        self.weights_plif.append(layer.neuron.b_param)
    
    def reset_model(self, train=True):
        """Reset model state."""
        functional.reset_net(self)
        for i, layer in enumerate(self.snn):
            if self.config.sparsity_p > 0 and i < len(self.mask):
                with torch.no_grad():
                    if hasattr(layer, 'W'):
                        self.mask[i] = self.mask[i].to(layer.W.weight.device)
                        layer.W.weight *= self.mask[i]
    
    def decrease_sig(self, epoch):
        """Placeholder for sigma decrease (not used in SNN)."""
        pass
    
    def forward(self, x):
        """
        Forward pass following SSM-inspired-LIF pattern.
        
        Input shape: (time, batch, features) - spikingjelly step_mode='m' format
        Output shape: (time, batch, features) or (time, batch, features) for v_seq
        """
        avg_spikes = []
        
        # Process all layers - similar to SSM-inspired-LIF forward
        for i, snn_lay in enumerate(self.snn):
            if i == len(self.snn) - 1:  # Output layer
                x = snn_lay(x)
                # Return v_seq if not using spike_count loss
                if self.config.loss != 'spike_count' and hasattr(snn_lay.neuron, 'v_seq'):
                    x = snn_lay.neuron.v_seq
            else:  # Hidden layers
                x = snn_lay(x)
                # Track average spikes from layer's stored spikes (before dropout)
                if hasattr(snn_lay, '_last_spikes'):
                    avg_spikes.append(snn_lay._last_spikes.sum(dim=(0, 1)).mean())
                else:
                    # Fallback: approximate from output
                    avg_spikes.append((x > 0).float().sum(dim=(0, 1)).mean())
        
        return x, avg_spikes
    
    def get_model_wandb_logs(self):
        """Get model logs for wandb."""
        model_logs = {"sigma": 0}
        
        for i, layer in enumerate(self.snn):
            # Membrane time constant
            if hasattr(layer, 'neuron') and isinstance(layer.neuron, nn.Module):
                if hasattr(layer.neuron, 'tau'):
                    tau_m = layer.neuron.tau
                elif hasattr(layer.neuron, 'w'):
                    tau_m = 1. / layer.neuron.w.sigmoid()
                else:
                    tau_m = 0
            else:
                tau_m = 0
            
            # Synapse time constant
            if hasattr(layer, 'synapse_filter') and layer.use_stateful_synapse:
                if hasattr(layer.synapse_filter, 'tau'):
                    tau_s = layer.synapse_filter.tau
                elif hasattr(layer.synapse_filter, 'w'):
                    tau_s = 1. / layer.synapse_filter.w.sigmoid()
                else:
                    tau_s = 0
            else:
                tau_s = 0
            
            # Weight magnitude
            if hasattr(layer, 'W'):
                w = torch.abs(layer.W.weight).mean()
            else:
                w = 0
            
            model_logs.update({
                f'tau_m_{i}': tau_m * self.config.time_step,
                f'tau_s_{i}': tau_s * self.config.time_step,
                f'w_{i}': w
            })
        
        return model_logs

