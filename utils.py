"""
Utility functions for SNN models with delays.

Modified from:
https://github.com/Thvnvtos/SNN-delays

MIT License
Copyright (c) 2025
"""

import numpy as np
import random, sys
import torch
import yaml
import os
from spikingjelly.activation_based import surrogate


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params_with_sparsity(model, sparsity_level):
    total = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.endswith(".P") or name.endswith(".SIG"):
            total += int(p.numel() * (1 - sparsity_level))
        else:
            total += p.numel()

    return total


def check_versions():
    python_version = sys.version .split(' ')[0]
    print("============== Checking Packages versions ================")
    print(f"python {python_version}")
    print(f"numpy {np.__version__}")
    print(f"pytorch {torch.__version__}")



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False

    # I don't know if this is useful, look it up.
    #os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Config loading functionality
# ============================================================================

class Config:
    """Config class that loads from YAML and computes derived values."""
    
    def __init__(self, yaml_path=None, config_dict=None):
        """
        Initialize config from YAML file or dictionary.
        
        Args:
            yaml_path: Path to YAML config file
            config_dict: Dictionary of config values (alternative to yaml_path)
        """
        if yaml_path:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_dict is None:
            raise ValueError("Either yaml_path or config_dict must be provided")
        
        # Set all attributes from YAML
        for key, value in config_dict.items():
            # Handle values format: if value is a dict with 'values' key, extract first value
            if isinstance(value, dict) and 'values' in value:
                if len(value['values']) > 0:
                    setattr(self, key, value['values'][0])
                else:
                    setattr(self, key, None)
            else:
                setattr(self, key, value)
        
        # Compute derived values
        self._compute_derived_values()
    
    def _compute_derived_values(self):
        """Compute all derived values that depend on other config values."""
        # n_inputs
        if not hasattr(self, 'n_inputs') or self.n_inputs is None:
            self.n_inputs = 700 // self.n_bins
        
        # n_outputs
        if not hasattr(self, 'n_outputs') or self.n_outputs is None:
            self.n_outputs = 20 if self.dataset == 'shd' else 35
        
        # output_v_threshold
        if not hasattr(self, 'output_v_threshold') or self.output_v_threshold is None:
            self.output_v_threshold = 2.0 if self.loss == 'spike_count' else 1e9
        
        # surrogate_function
        if not hasattr(self, 'surrogate_function') or self.surrogate_function is None:
            if hasattr(self, 'surrogate_type'):
                if self.surrogate_type == 'ATan':
                    self.surrogate_function = surrogate.ATan(alpha=self.alpha)
                elif self.surrogate_type == 'ATanThreshold':
                    beta = getattr(self, 'beta', 1.0)
                    self.surrogate_function = surrogate.ATanThreshold(alpha=self.alpha, beta=beta)
                elif self.surrogate_type == 'BoxcarThreshold':
                    x_min = getattr(self, 'x_min', None)
                    x_max = getattr(self, 'x_max', None)
                    self.surrogate_function = surrogate.BoxcarThreshold(
                        threshold=self.v_threshold, x_min=x_min, x_max=x_max
                    )
            else:
                # Default to ATan
                self.surrogate_function = surrogate.ATan(alpha=self.alpha)
        
        # surrogate_function_output
        if not hasattr(self, 'surrogate_function_output') or self.surrogate_function_output is None:
            self.surrogate_function_output = surrogate.ATan(alpha=self.alpha)
        
        # init_tau (convert from ms to time steps)
        # If init_tau_ms is provided, use it; otherwise use init_tau if it's a raw value
        if hasattr(self, 'init_tau_ms'):
            init_tau_raw = self.init_tau_ms
        elif hasattr(self, 'init_tau') and isinstance(self.init_tau, (int, float)) and self.init_tau > 1:
            # If init_tau is a large number (> 1), assume it's in ms
            init_tau_raw = self.init_tau
        else:
            init_tau_raw = 15.0  # Default
        
        self.init_tau = (init_tau_raw + 1e-9) / self.time_step
        
        # stateful_synapse_tau (convert from ms to time steps)
        if hasattr(self, 'stateful_synapse_tau_ms'):
            stateful_synapse_tau_raw = self.stateful_synapse_tau_ms
        elif hasattr(self, 'stateful_synapse_tau') and isinstance(self.stateful_synapse_tau, (int, float)) and self.stateful_synapse_tau > 1:
            # If stateful_synapse_tau is a large number (> 1), assume it's in ms
            stateful_synapse_tau_raw = self.stateful_synapse_tau
        else:
            stateful_synapse_tau_raw = 10.0  # Default
        
        self.stateful_synapse_tau = (stateful_synapse_tau_raw + 1e-9) / self.time_step
        
        # max_delay
        if not hasattr(self, 'max_delay') or self.max_delay is None:
            if hasattr(self, 'max_delay_ms'):
                self.max_delay = self.max_delay_ms // self.time_step
            else:
                # Default calculation
                self.max_delay = 300 // self.time_step
        
        # Ensure max_delay is odd
        if self.max_delay % 2 == 0:
            self.max_delay = self.max_delay + 1
        
        # Delay-related computed values
        if not hasattr(self, 'sigInit') or self.sigInit is None:
            self.sigInit = self.max_delay // 2 if self.model_type == 'snn_delays' else 0
        
        if not hasattr(self, 'final_epoch') or self.final_epoch is None:
            self.final_epoch = (1 * self.epochs) // 4 if self.model_type == 'snn_delays' else 0
        
        if not hasattr(self, 'left_padding') or self.left_padding is None:
            self.left_padding = self.max_delay - 1
        
        if not hasattr(self, 'right_padding') or self.right_padding is None:
            self.right_padding = (self.max_delay - 1) // 2
        
        if not hasattr(self, 'init_pos_a') or self.init_pos_a is None:
            self.init_pos_a = -self.max_delay // 2
        
        if not hasattr(self, 'init_pos_b') or self.init_pos_b is None:
            self.init_pos_b = self.max_delay // 2
        
        # DCLSversion
        if not hasattr(self, 'DCLSversion') or self.DCLSversion is None:
            self.DCLSversion = 'gauss' if self.model_type == 'snn_delays' else 'max'
        
        # lr_pos
        if not hasattr(self, 'lr_pos') or self.lr_pos is None:
            self.lr_pos = 100 * self.lr_w if self.model_type == 'snn_delays' else 0
        
        # scheduler_pos
        if not hasattr(self, 'scheduler_pos') or self.scheduler_pos is None:
            self.scheduler_pos = 'cosine_a' if self.model_type == 'snn_delays' else 'none'
        
        # max_lr_w
        if not hasattr(self, 'max_lr_w') or self.max_lr_w is None:
            self.max_lr_w = 5 * self.lr_w
        
        # max_lr_pos
        if not hasattr(self, 'max_lr_pos') or self.max_lr_pos is None:
            self.max_lr_pos = 5 * self.lr_pos
        
        # t_max_w
        if not hasattr(self, 't_max_w') or self.t_max_w is None:
            self.t_max_w = self.epochs
        
        # t_max_pos
        if not hasattr(self, 't_max_pos') or self.t_max_pos is None:
            self.t_max_pos = self.epochs
        
        # max_lr_w_finetuning
        if not hasattr(self, 'max_lr_w_finetuning') or self.max_lr_w_finetuning is None:
            if hasattr(self, 'lr_w_finetuning'):
                self.max_lr_w_finetuning = 1.2 * self.lr_w_finetuning
        
        # time_mask_size
        if not hasattr(self, 'time_mask_size') or self.time_mask_size is None:
            self.time_mask_size = self.max_delay // 3
        
        # neuron_mask_size
        if not hasattr(self, 'neuron_mask_size') or self.neuron_mask_size is None:
            self.neuron_mask_size = self.n_inputs // 5
        
        # wandb run info strings
        if not hasattr(self, 'run_info') or self.run_info is None:
            self.run_info = f'||{self.model_type}||{self.dataset}||{self.time_step}ms||bins={self.n_bins}'
        
        # Default run_name if not provided
        if not hasattr(self, 'run_name') or self.run_name is None:
            self.run_name = self.dataset.upper()
        
        # Default wandb_project_name if not provided
        if not hasattr(self, 'wandb_project_name') or self.wandb_project_name is None:
            self.wandb_project_name = 'SNNDelays'


def load_config(config_path):
    """
    Load config from YAML file or Python module.
    
    Args:
        config_path: Path to config file (YAML) or module name (Python)
    
    Returns:
        Config object
    """
    # Check if it's a YAML file
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        if os.path.exists(config_path):
            return Config(yaml_path=config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Check if it's a path to a YAML file in configs directory
    if config_path.startswith('configs.'):
        config_name = config_path.replace('configs.', '')
        yaml_path = os.path.join('configs', f'{config_name}.yaml')
        if os.path.exists(yaml_path):
            return Config(yaml_path=yaml_path)
    
    # Check if it's a direct path to a YAML file
    if '/' in config_path and (config_path.endswith('.yaml') or config_path.endswith('.yml')):
        if os.path.exists(config_path):
            return Config(yaml_path=config_path)
    
    # Fallback to Python module import (for backward compatibility with config.py)
    import importlib
    try:
        config_module = importlib.import_module(config_path)
        ConfigClass = config_module.Config
        return ConfigClass()
    except (ImportError, AttributeError) as e:
        raise Exception(f'Failed to load config from {config_path}: {e}')