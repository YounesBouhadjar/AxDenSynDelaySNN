# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Layer classes for SNN models with delays.
Each layer is a complete, self-contained nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer
from DCLS.construct.modules import Dcls1d


class DelayedMaskedDcls1d(Dcls1d):
    """Delayed DCLS layer with scheduled sparsity mask."""
    def __init__(
        self,
        *args,
        total_epochs,
        final_sparsity=0.8,
        warmup_fraction=0.25,
        learned_mask=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.total_epochs = total_epochs
        self.Si = 0.0
        self.Sf = final_sparsity
        self.tf = int(warmup_fraction * total_epochs)
        self.t = 0
        self.learned_mask = learned_mask
        
        if learned_mask:
            self.delay_scores = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            self.delay_scores = (torch.rand(self.out_channels) > self.Sf).float()
    
    def update_epochs(self, epoch):
        self.t = epoch
    
    def scheduled_sparsity(self):
        if self.learned_mask:
            if self.t >= self.tf:
                return self.Sf
            frac = self.t / self.tf
            return self.Sf - (self.Sf - self.Si) * (1 - frac) ** 3
        else:
            return self.Sf
    
    def forward(self, x):
        y = super().forward(x)
        if self.learned_mask:
            S_t = self.scheduled_sparsity()
            scores = self.delay_scores
            prob = torch.sigmoid(scores)
            k = int((1.0 - S_t) * prob.numel())
            k = max(1, min(k, prob.numel()))
            thresh = torch.topk(prob, k=k).values.min()
            hard = (prob >= thresh).float()
            m = hard + (prob - prob.detach())
        else:
            m = self.delay_scores.to(x.device)
        
        m = m.view(1, -1, 1)
        Cin = x.shape[2]
        Cout = y.shape[2]
        if Cin != Cout:
            pad = Cin - Cout
            y = torch.nn.functional.pad(y, (0, pad))
        return m * y + (1.0 - m) * x


class MaskedDcls1d(Dcls1d):
    """DCLS layer with fixed sparsity mask."""
    def __init__(
        self,
        *args,
        sparsity_p_delay=0.,
        learned_mask=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sparsity_p_delay = sparsity_p_delay
        self.learned_mask = learned_mask
        self.n_inputs = args[0]
        
        if self.learned_mask:
            self.m = torch.nn.Parameter(torch.randn(self.n_inputs))
        else:
            self.m = (torch.rand(self.n_inputs) > self.sparsity_p_delay).float()
    
    def forward(self, x):
        y = super().forward(x)
        if self.learned_mask:
            prob = torch.sigmoid(self.m)
            k = int((1.0 - self.sparsity_p_delay) * prob.numel())
            k = max(1, min(k, prob.numel()))
            thresh = torch.topk(prob, k=k).values.min()
            hard = (prob >= thresh).float()
            m = hard + (prob - prob.detach())
        else:
            m = self.m
        m = m.view(1, -1, 1)
        Cin = x.shape[2]
        Cout = y.shape[2]
        if Cin != Cout:
            assert Cin > Cout, (Cin, Cout)
            pad = Cin - Cout
            y = torch.nn.functional.pad(y, (0, pad))
        return m * y + (1.0 - m) * x


class LIFLayer(nn.Module):
    """
    A single layer of Leaky Integrate-and-Fire neurons.
    Similar structure to SiLIFLayer but using spikingjelly components.
    
    Arguments
    ---------
    input_size : int
        Number of input features
    hidden_size : int
        Number of output neurons
    config : object
        Configuration object with all model parameters
    is_output_layer : bool
        Whether this is the output layer (affects neuron type and threshold)
    """
    
    def __init__(
        self,
        input_size,
        hidden_size,
        config,
        is_output_layer=False
    ):
        super().__init__()
        
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.config = config
        self.is_output_layer = is_output_layer
        
        # Linear transformation
        self.W = layer.Linear(
            self.input_size, 
            self.hidden_size, 
            bias=config.bias, 
            step_mode='m'
        )
        
        # Batch normalization
        self.normalize = False
        if config.use_batchnorm and not is_output_layer:
            self.norm = layer.BatchNorm1d(self.hidden_size, step_mode='m')
            self.normalize = True
        
        # Neuron type
        neuron_type = config.spiking_neuron_output_type if is_output_layer else config.spiking_neuron_type
        threshold = config.output_v_threshold if is_output_layer else config.v_threshold
        surrogate = config.surrogate_function_output if is_output_layer else config.surrogate_function
        
        if neuron_type == 'lif':
            self.neuron = neuron.LIFNode(
                tau=config.init_tau,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True
            )
        elif neuron_type == 'plif':
            self.neuron = neuron.ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True
            )
        elif neuron_type == 'silif':
            self.neuron = neuron.SiLIFNode(
                tau=config.init_tau,
                nb_neurons=self.hidden_size,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                v_reset=getattr(config, 'v_reset', None),
                step_mode='m',
                decay_input=False,
                store_v_seq=False
            )
        elif neuron_type == 'heaviside':
            self.neuron = surrogate
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        # Dropout
        self.drop = layer.Dropout(config.dropout_p, step_mode='m')
        
        # Stateful synapse
        self.use_stateful_synapse = config.stateful_synapse and not is_output_layer
        if self.use_stateful_synapse:
            self.synapse_filter = layer.SynapseFilter(
                tau=config.stateful_synapse_tau,
                learnable=config.stateful_synapse_learnable,
                step_mode='m'
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Input shape: (time, batch, features) - spikingjelly step_mode='m' format
        Output shape: (time, batch, features)
        
        Returns processed output (after dropout and synapse filter).
        Spikes can be accessed via self.neuron if needed.
        """
        # Linear transformation
        x = self.W(x)
        
        # Batch normalization
        if self.normalize:
            x = self.norm(x.unsqueeze(3)).squeeze()
        
        # Neuron dynamics
        if self.config.spiking_neuron_type == 'heaviside':
            spikes = self.neuron(x - self.config.v_threshold)
        else:
            spikes = self.neuron(x)
        
        # Store spikes for tracking (before dropout)
        self._last_spikes = spikes
        
        # Dropout
        x = self.drop(spikes)
        
        # Stateful synapse
        if self.use_stateful_synapse:
            x = self.synapse_filter(x)
        
        return x


class DelayLayer(nn.Module):
    """
    A single layer with delay processing (DCLS) and LIF neurons.
    Similar structure to SiLIFLayer but adapted for delay-based SNNs.
    
    Arguments
    ---------
    input_size : int
        Number of input features
    hidden_size : int
        Number of output neurons
    config : object
        Configuration object with all model parameters
    is_output_layer : bool
        Whether this is the output layer
    """
    
    def __init__(
        self,
        input_size,
        hidden_size,
        config,
        is_output_layer=False,
    ):
        super().__init__()
        
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.config = config
        self.is_output_layer = is_output_layer
        self.delay_type = getattr(config, 'delay_type', 'synaptic')

        if self.delay_type == 'dendritic':
            self.W = nn.Linear(
                self.input_size,
                self.hidden_size,
                bias=config.bias_layer
            )

        # Delay layer (DCLS)
        self._init_delay_layer()

        if self.delay_type == 'axonal':
            self.W = nn.Linear(
                self.input_size,
                self.hidden_size,
                bias=config.bias_layer
            )
 
        # Batch normalization
        self.normalize = False
        if config.use_batchnorm and not is_output_layer:
            self.norm = layer.BatchNorm1d(self.hidden_size, step_mode='m')
            self.normalize = True
        
        # Output layer normalization
        if is_output_layer and getattr(config, 'use_batchnorm_output', False):
            self.norm_output = nn.LayerNorm(self.hidden_size)
            self.normalize_output = True
        else:
            self.normalize_output = False
        
        # Neuron type
        neuron_type = config.spiking_neuron_output_type if is_output_layer else config.spiking_neuron_type
        threshold = config.output_v_threshold if is_output_layer else config.v_threshold
        surrogate = config.surrogate_function_output if is_output_layer else config.surrogate_function
        
        if neuron_type == 'lif':
            self.neuron = neuron.LIFNode(
                tau=config.init_tau,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True
            )
        elif neuron_type == 'plif':
            self.neuron = neuron.ParametricLIFNode(
                init_tau=config.init_tau,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                step_mode='m',
                decay_input=False,
                store_v_seq=True
            )
        elif neuron_type == 'silif':
            self.neuron = neuron.SiLIFNode(
                tau=config.init_tau,
                nb_neurons=self.hidden_size,
                v_threshold=threshold,
                surrogate_function=surrogate,
                detach_reset=config.detach_reset,
                v_reset=getattr(config, 'v_reset', None),
                step_mode='m',
                decay_input=False,
                store_v_seq=False
            )
        elif neuron_type == 'heaviside':
            self.neuron = surrogate
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
        
        # Dropout
        self.drop = layer.Dropout(config.dropout_p, step_mode='m')
        
        # Stateful synapse
        self.use_stateful_synapse = config.stateful_synapse and not is_output_layer
        if self.use_stateful_synapse:
            self.synapse_filter = layer.SynapseFilter(
                tau=config.stateful_synapse_tau,
                learnable=config.stateful_synapse_learnable,
                step_mode='m'
            )
    
    def _init_delay_layer(self):
        """Initialize the delay (DCLS) layer based on delay type."""
        if self.delay_type == 'axonal':
            if getattr(self.config, 'sparsity_p_delay', 0) > 0:
                self.delay_layer = DelayedMaskedDcls1d(
                    self.input_size,
                    self.input_size,
                    kernel_count=1,
                    groups=self.input_size,
                    version='gauss',
                    bias=self.config.bias,
                    total_epochs=self.config.epochs,
                    dilated_kernel_size=self.config.max_delay,
                    final_sparsity=self.config.sparsity_p_delay,
                    learned_mask=getattr(self.config, 'learned_mask', False)
                )
            else:
                self.delay_layer = Dcls1d(
                    self.input_size,
                    self.input_size,
                    kernel_count=1,
                    groups=self.input_size,
                    version='gauss',
                    bias=self.config.bias,
                    dilated_kernel_size=self.config.max_delay
                )
            self.delay_layer.weight.requires_grad = False
            self.delay_layer.weight.fill_(1.)
        elif self.delay_type == 'dendritic':
            assert getattr(self.config, 'sparsity_p_delay', 0) == 0, "delay sparsity is not supported yet for dendritic delay"
            self.delay_layer = Dcls1d(
                self.hidden_size,
                self.hidden_size,
                kernel_count=1,
                version='gauss',
                bias=self.config.bias,
                groups=self.hidden_size,
                dilated_kernel_size=self.config.max_delay
            )
            self.delay_layer.weight.requires_grad = False
            self.delay_layer.weight.fill_(1.)
        else:
            assert getattr(self.config, 'sparsity_p_delay', 0) == 0, "delay sparsity is not supported yet for synaptic delay"
            self.delay_layer = Dcls1d(
                self.input_size,
                self.hidden_size,
                kernel_count=self.config.kernel_count,
                groups=1,
                dilated_kernel_size=self.config.max_delay,
                bias=self.config.bias,
                version=self.config.DCLSversion
            )
    
    def forward(self, x):
        """
        Forward pass with delay processing.
        
        Input shape: (time, batch, features) - spikingjelly step_mode='m' format
        Output shape: (time, batch, features)
        """
        # Apply delay processing based on delay_type
        if self.delay_type == 'dendritic':
            x = F.pad(x, (0, 0, 0, 0, self.config.left_padding, self.config.right_padding), 'constant', 0)
            x = self.W(x)
            # Permute for DCLS: (time, batch, features) -> (batch, features, time)
            x = x.permute(1, 2, 0)
            x = self.delay_layer(x)
            # Permute back: (batch, features, time) -> (time, batch, features)
            x = x.permute(2, 0, 1)
        else:  # axonal or synaptic
            # Permute for DCLS: (time, batch, features) -> (batch, features, time)
            x = x.permute(1, 2, 0)
            x = F.pad(x, (self.config.left_padding, self.config.right_padding), 'constant', 0)
            x = self.delay_layer(x)
            # Permute back: (batch, features, time) -> (time, batch, features)
            x = x.permute(2, 0, 1)
            # Linear transformation for axonal (after delay)
            if self.delay_type == 'axonal':
                x = self.W(x)
        
        # Batch normalization (only for hidden layers)
        if self.normalize:
            x = self.norm(x.unsqueeze(3)).squeeze()
        
        # Output layer normalization (applied BEFORE neuron, like old code)
        if self.is_output_layer and self.normalize_output:
            x = self.norm_output(x)
        
        # Store input for voltage regularization (before neuron)
        x_input = x.clone() if getattr(self.config, 'use_voltage_reg', False) else None
        
        # Neuron dynamics
        if self.config.spiking_neuron_type == 'heaviside':
            spikes = self.neuron(x - self.config.v_threshold)
        else:
            spikes = self.neuron(x)
        
        # Store spikes for tracking (before dropout)
        self._last_spikes = spikes
        
        # Output layer: skip dropout and synapse filter, return spikes directly
        if self.is_output_layer:
            x = spikes
        else:
            # Hidden layers: apply dropout and synapse filter
            # Dropout
            x = self.drop(spikes)
            
            # Stateful synapse
            if self.use_stateful_synapse:
                x = self.synapse_filter(x)
        
        return x, spikes, x_input
