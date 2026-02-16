# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Spiking Neural Network with learnable delays (axonal, dendritic, synaptic).

This code is modified from:
https://github.com/Thvnvtos/SNN-delays
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional
from DCLS.construct.modules import Dcls1d

from train import Model
from utils import set_seed
from models.layers import DelayLayer


class SnnDelays(Model):
    """
    Multi-layered Spiking Neural Network with delays using layer-based structure.
    Similar to SSM-inspired-LIF SNN class but adapted for delay-based SNNs with spikingjelly.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.delay_type = getattr(config, 'delay_type', 'synaptic')
    
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
        
        # First layer
        snn.append(
            DelayLayer(
                input_size=input_size,
                hidden_size=self.config.n_hidden_neurons,
                config=self.config,
            )
        )
        input_size = self.config.n_hidden_neurons
        
        # Hidden layers
        for i in range(self.config.n_hidden_layers - 1):
            snn.append(
                DelayLayer(
                    input_size=input_size,
                    hidden_size=self.config.n_hidden_neurons,
                    config=self.config,
                )
            )
        
        # Output layer
        snn.append(
            DelayLayer(
                input_size=input_size,
                hidden_size=self.config.n_outputs,
                config=self.config,
                is_output_layer=True,
            )
        )
        
        return snn
    
    def init_model(self):
        """Initialize model weights and parameters."""
        set_seed(self.config.seed)
        self.mask = []
        
        # Initialize linear layers for axonal/dendritic delays
        for layer in self.snn:
            if hasattr(layer, 'W'):
                torch.nn.init.kaiming_uniform_(layer.W.weight, nonlinearity='relu')
                if self.config.sparsity_p > 0:
                    with torch.no_grad():
                        mask = (torch.rand(layer.W.weight.size()) > self.config.sparsity_p).float()
                        layer.W.weight *= mask
                        self.mask.append(mask)
             
        # Initialize delay layers (DCLS) for synaptic delays
        if self.config.init_w_method == 'kaiming_uniform' and self.config.delay_type == 'synaptic':
            for layer in self.snn:
                if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                    torch.nn.init.kaiming_uniform_(layer.delay_layer.weight, nonlinearity='relu')
                    if self.config.sparsity_p > 0:
                        with torch.no_grad():
                            mask = torch.rand(layer.delay_layer.weight.size()).to(layer.delay_layer.weight.device)
                            mask[mask > self.config.sparsity_p] = 1
                            mask[mask <= self.config.sparsity_p] = 0
                            layer.delay_layer.weight *= mask
                            self.mask.append(mask)
        
        # Initialize delay positions
        if self.config.init_pos_method == 'uniform':
            for layer in self.snn:
                if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                    torch.nn.init.uniform_(layer.delay_layer.P, a=self.config.init_pos_a, b=self.config.init_pos_b)
                    layer.delay_layer.clamp_parameters()
                    if self.config.model_type == 'snn_delays_lr0':
                        layer.delay_layer.P.requires_grad = False
        
        # Initialize delay sigma
        for layer in self.snn:
            if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                torch.nn.init.constant_(layer.delay_layer.SIG, self.config.sigInit)
                layer.delay_layer.SIG.requires_grad = False
        
        # Collect parameters for optimization (needed for Model base class)
        self.positions = []
        self.weights = []
        self.weights_bn = []
        self.weights_plif = []
        
        for layer in self.snn:
            # Delay layer positions and weights
            if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                self.positions.append(layer.delay_layer.P)
                self.weights.append(layer.delay_layer.weight)
                if self.config.bias:
                    self.weights_bn.append(layer.delay_layer.bias)
            
            # Linear layer weights for axonal/dendritic delays
            if hasattr(layer, 'W'):
                self.weights.append(layer.W.weight)
            
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
        
        # Store initial positions for reference
        self.init_pos = []
        for layer in self.snn:
            if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                self.init_pos.append(layer.delay_layer.P.cpu().detach().numpy().copy())
    
    def reset_model(self, train=True):
        """Reset model state."""
        functional.reset_net(self)
        k = 0
        
        # Reset linear layers and apply sparsity masks
        for layer in self.snn:
            if hasattr(layer, 'W'):
                functional.reset_net(layer.W)
                if self.config.sparsity_p > 0 and k < len(self.mask):
                    with torch.no_grad():
                        self.mask[k] = self.mask[k].to(layer.W.weight.device)
                        layer.W.weight *= self.mask[k]
                        k += 1

        # Reset delay layers and apply sparsity masks
        if self.config.sparsity_p > 0 and self.config.delay_type == 'synaptic':
            for layer in self.snn:
                if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d) and k < len(self.mask):
                    with torch.no_grad():
                        self.mask[k] = self.mask[k].to(layer.delay_layer.weight.device)
                        layer.delay_layer.weight *= self.mask[k]
                        k += 1
        
        # Clamp delay parameters
        if train:
            for layer in self.snn:
                if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                    layer.delay_layer.clamp_parameters()
    
    def decrease_sparsity(self, epoch):
        """Update sparsity schedule for delay layers."""
        for layer in self.snn:
            if hasattr(layer, 'delay_layer') and hasattr(layer.delay_layer, 'update_epochs'):
                layer.delay_layer.update_epochs(epoch)
    
    def decrease_sig(self, epoch):
        """Decrease sigma for delay layers."""
        alpha = 0
        # Get sigma from last layer
        if hasattr(self.snn[-1], 'delay_layer') and isinstance(self.snn[-1].delay_layer, Dcls1d):
            sig = self.snn[-1].delay_layer.SIG[0, 0, 0, 0].detach().cpu().item()
            
            if self.config.decrease_sig_method == 'exp':
                if epoch < self.config.final_epoch and sig > 0.23:
                    if self.config.DCLSversion == 'max':
                        alpha = (1 / self.config.sigInit) ** (1 / (self.config.final_epoch))
                    elif self.config.DCLSversion == 'gauss':
                        alpha = (0.23 / self.config.sigInit) ** (1 / (self.config.final_epoch))
                    
                    for layer in self.snn:
                        if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                            layer.delay_layer.SIG *= alpha
    
    def round_pos(self):
        """Round delay positions to integers."""
        with torch.no_grad():
            for layer in self.snn:
                if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                    layer.delay_layer.P.round_()
                    layer.delay_layer.clamp_parameters()
    
    def forward(self, x):
        """
        Forward pass following SSM-inspired-LIF pattern.
        
        Input shape: (time, batch, features) - spikingjelly step_mode='m' format
        Output shape: (time, batch, features) or (time, batch, features) for v_seq
        Returns: (output, [avg_spikes, firing_rates, ops, voltage_reg, max_population_frs])
        """
        avg_spikes = []
        firing_rates = []
        ops = []
        max_population_frs = []
        voltage_reg_values = []
        
        # Process all layers
        for i, snn_lay in enumerate(self.snn):
            # Track presynaptic spikes for ops calculation
            nb_presynaptic_spikes = (x != 0).sum(dim=(0, 2), dtype=float).mean()

            # Track max population firing rates
            if self.config.model_type == 'snn_delays':
                window_avg = (x > 0.0).to(x.dtype).unfold(0, self.config.max_delay, 1).mean(dim=(-1, -2))
                max_population_frs.append(window_avg.max())
            else:
                window_avg = (x > 0.0).to(x.dtype).unfold(0, getattr(self.config, 'fr_window_size', 10), 1).mean(dim=(-1, -2))
                max_population_frs.append(window_avg.max())
            
            # Forward through layer
            x, spikes, x_input = snn_lay(x)
            
            # Voltage regularization
            if getattr(self.config, 'use_voltage_reg', False) and self.config.spiking_neuron_type != 'heaviside':
                if hasattr(snn_lay.neuron, 'v_seq') and snn_lay.neuron.v_seq is not None and x_input is not None:
                    v_seq = snn_lay.neuron.v_seq
                    voltage_sum = v_seq + x_input
                    voltage_reg = F.relu(voltage_sum - self.config.voltage_reg_threshold).sum()
                    voltage_reg_values.append(voltage_reg)
                else:
                    voltage_reg_values.append(torch.tensor(0.0, device=x.device))
            else:
                voltage_reg_values.append(torch.tensor(0.0, device=x.device))
            
            # Track spikes and firing rates
            if i < len(self.snn) - 1:  # Hidden layers
                avg_spikes.append((spikes != 0).sum(dim=(0, 2), dtype=float).mean())
                firing_rates.append(spikes.mean(dim=(0, 1)))
                ops.append(nb_presynaptic_spikes * self.config.n_hidden_neurons)
            else:  # Output layer
                ops.append(nb_presynaptic_spikes * self.config.n_outputs)
        
        # Final max population firing rate
        if self.config.model_type == 'snn_delays':
            window_avg = x.unfold(0, self.config.max_delay, 1).mean(dim=(-1, -2))
            max_population_frs.append(window_avg.max())
        else:
            window_avg = x.unfold(0, getattr(self.config, 'fr_window_size', 10), 1).mean(dim=(-1, -2))
            max_population_frs.append(window_avg.max())
        
        # Return v_seq if not using spike_count loss
        if self.config.loss != 'spike_count' and hasattr(self.snn[-1].neuron, 'v_seq'):
            x = self.snn[-1].neuron.v_seq
        
        # Concatenate firing rates
        if firing_rates:
            firing_rates = torch.cat(firing_rates)
        else:
            firing_rates = torch.tensor([])
        
        voltage_reg_total = torch.stack(voltage_reg_values).sum() if voltage_reg_values else torch.tensor(0.0)
        
        return x, [
            torch.tensor(avg_spikes),
            firing_rates,
            torch.tensor(ops),
            voltage_reg_total,
            torch.stack(max_population_frs)
        ]
    
    def get_model_wandb_logs(self):
        """Get model logs for wandb."""
        # Get sigma from last delay layer
        sig = 0
        if hasattr(self.snn[-1], 'delay_layer') and isinstance(self.snn[-1].delay_layer, Dcls1d):
            sig = self.snn[-1].delay_layer.SIG[0, 0, 0, 0].detach().cpu().item()
        
        model_logs = {"sigma": sig}
        
        for i, layer in enumerate(self.snn):
            # Membrane time constant
            if hasattr(layer, 'neuron') and isinstance(layer.neuron, nn.Module):
                if self.config.spiking_neuron_type in ['lif', 'lif_ex']:
                    tau_m = layer.neuron.tau
                elif self.config.spiking_neuron_type == 'silif':
                    tau_m = layer.neuron.tau
                elif self.config.spiking_neuron_type == 'plif':
                    tau_m = 1. / layer.neuron.w.sigmoid()
                else:
                    tau_m = 0
            else:
                tau_m = 0
            
            # Synapse time constant
            if hasattr(layer, 'synapse_filter') and layer.use_stateful_synapse and i < len(self.snn) - 1:
                if hasattr(layer.synapse_filter, 'tau'):
                    tau_s = layer.synapse_filter.tau
                elif hasattr(layer.synapse_filter, 'w'):
                    tau_s = 1. / layer.synapse_filter.w.sigmoid()
                else:
                    tau_s = 0
            else:
                tau_s = 0
            
            # Weight magnitude
            if hasattr(layer, 'delay_layer') and isinstance(layer.delay_layer, Dcls1d):
                w = torch.abs(layer.delay_layer.weight).mean()
            elif hasattr(layer, 'W'):
                w = torch.abs(layer.W.weight).mean()
            else:
                w = 0
            
            model_logs.update({
                f'tau_m_{i}': tau_m * self.config.time_step,
                f'tau_s_{i}': tau_s * self.config.time_step,
                f'w_{i}': w
            })
        
        return model_logs

