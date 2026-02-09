# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Main entry point for training SNN models with delays.
"""

import torch
import wandb
import os
import numpy as np
import importlib
from utils import set_seed
from config_loader import load_config

from datasets import SHD_dataloaders, SSC_dataloaders, GSC_dataloaders, SHD_NORM_dataloaders, SSC_NORM_dataloaders
from models.snn_delays import SnnDelays
from models.snn import SNN
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n=====> Device = {device} \n\n")

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="SNN Delays Training Script")
parser.add_argument('--config', type=str, default=None, help='Config module name (e.g., configs.SHD, config). If not specified, will use dataset to infer.')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--delay_type', type=str, default=None, help='Delay type')
parser.add_argument('--dataset', type=str, default=None, help='Dataset')
parser.add_argument('--n_hidden_layers', type=int, default=None, help='Number of hidden layers')
parser.add_argument('--v_threshold', type=float, default=None, help='Threshold')
parser.add_argument('--reg_factor', type=float, default=None, help='Regularization factor')
parser.add_argument('--reg_fmin', type=float, default=None, help='Regularization fmin')
parser.add_argument('--reg_fmax', type=float, default=None, help='Regularization fmax')
parser.add_argument('--test_pretrained_model', action="store_true")
parser.add_argument('--run_name', type=str, default='Test_Runs', help='Run name')
parser.add_argument('--loss', type=str, default=None, help='Loss')
parser.add_argument('--conv_bn_penalty', type=str2bool, default=None, help='Conv BN Penalty')
parser.add_argument('--bias_layer', type=str2bool, default=None, help='Bias Layer')
parser.add_argument('--bn_penalty_weight', type=float, default=None, help='BN Penalty Weight')
###
parser.add_argument('--max_delay', type=int, default=None, help='Max delay')
parser.add_argument('--sigInit', type=float, default=None, help='Sigma Initial')
parser.add_argument('--final_epoch', type=int, default=None, help='Final epoch')
parser.add_argument('--init_tau', type=float, default=None, help='Initial tau')
parser.add_argument('--lr_w', type=float, default=None, help='Learning rate for weights')
parser.add_argument('--lr_pos', type=float, default=None, help='Learning rate for positions')
parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter for surrogate function')
parser.add_argument('--beta', type=float, default=None, help='Beta parameter for surrogate function')
parser.add_argument('--x_min', type=float, default=None, help='X min parameter for surrogate function')
parser.add_argument('--x_max', type=float, default=None, help='X max parameter for surrogate function')
parser.add_argument('--sparsity_p', type=float, default=None, help='Sparsity parameter for weights')
parser.add_argument('--sparsity_p_delay', type=float, default=None, help='Sparsity parameter for delay weights')
parser.add_argument('--epochs', type=float, default=None, help='Number of epochs')
parser.add_argument('--scheduler_w', type=str, default=None, help='Scheduler for weights')
parser.add_argument('--scheduler_pos', type=str, default=None, help='Scheduler for positions')
parser.add_argument('--use_regularizers', type=str2bool, default=None, help='Use regularizers')
#parser.add_argument('--time_step', type=int, default=None, help='Time step')
args = parser.parse_args()


# Load config from YAML
if args.config is None:
    if args.dataset is None:
        config_path = 'config'
    else:
        config_path = f'configs.{args.dataset}'

try:
    config = load_config(args.config)
except Exception as e:
    raise Exception(f'Failed to load config from {args.config}: {e}')

# Overwrite config parameters with command-line arguments if provided
# Overwrite any config attribute if provided as a command-line argument
for arg_key, arg_value in vars(args).items():
    if arg_value is not None and hasattr(config, arg_key):
        if arg_key == 'max_delay':
            arg_value = arg_value if arg_value%2==1 else arg_value+1
            setattr(config, 'sigInit', arg_value // 2)
            setattr(config, 'left_padding', arg_value-1)
            setattr(config, 'right_padding', (arg_value-1) // 2)
            setattr(config, 'init_pos_a', -arg_value//2)
            setattr(config, 'init_pos_b', arg_value//2)
        if arg_key == 'init_tau':
            arg_value = (arg_value  +  1e-9) / config.time_step
        if arg_key == 'lr_w':
            setattr(config, 'max_lr_w', 5 * arg_value)
        if arg_key == 'lr_pos':
            setattr(config, 'max_lr_pos', 5 * arg_value)
        if arg_key == 'alpha':
            from spikingjelly.activation_based import surrogate
            setattr(config, 'surrogate_function', surrogate.ATanThreshold(alpha = arg_value, beta=config.beta))
        if arg_key == 'beta':
            from spikingjelly.activation_based import surrogate
            setattr(config, 'surrogate_function', surrogate.ATanThreshold(alpha=config.alpha, beta = arg_value))
        if arg_key == 'x_min':
            from spikingjelly.activation_based import surrogate
            setattr(config, 'surrogate_function', surrogate.BoxcarThreshold(threshold = config.v_threshold, x_min = arg_value, x_max = config.x_max))
        if arg_key == 'x_max':
            from spikingjelly.activation_based import surrogate
            setattr(config, 'surrogate_function', surrogate.BoxcarThreshold(threshold = config.v_threshold, x_min = config.x_min, x_max = arg_value)) 
        if arg_key == 'epochs':
            arg_value = int(arg_value)
            final_epoch = (1*arg_value)//4
            setattr(config, 'final_epoch', final_epoch)
            setattr(config, 't_max_w', arg_value)
            setattr(config, 't_max_pos', arg_value)

        setattr(config, arg_key, arg_value)

if config.use_wandb:
    # Get all config attributes (class defaults + instance overrides) for wandb
    cfg = {k: v for k, v in config.__class__.__dict__.items() if not k.startswith('__') and not callable(v)}
    cfg.update({k: v for k, v in vars(config).items() if not k.startswith('__')})

    experiment_name = f'_seed_{config.seed}_{config.delay_type}_{config.dataset}_{config.time_step}ms_n_hidden_layers_{config.n_hidden_layers}_n_hidden_neurons_{config.n_hidden_neurons}'
    run_name = config.run_name + experiment_name
    config.save_model_path = run_name
    wandb.init(
        project= config.wandb_project_name,
        name=run_name,
        config = cfg)
        #group = self.config.wandb_group_name)
else:
    # If wandb is not used, set a default save path
    if not hasattr(config, 'save_model_path') or config.save_model_path is None:
        experiment_name = f'seed_{config.seed}_{config.delay_type}_{config.dataset}_{config.time_step}ms_n_hidden_layers_{config.n_hidden_layers}_n_hidden_neurons_{config.n_hidden_neurons}'
        config.save_model_path = config.run_name + experiment_name

# Create directory for saving model and config inside ckpt_models
config.save_model_path = os.path.join('ckpt_models', config.save_model_path)
os.makedirs(config.save_model_path, exist_ok=True)

# Save config as config.npy (NumPy format)
def save_config(config, filepath):
    """Save config object (class instance) as a NumPy .npy file."""
    # Convert config class instance to dict, including both instance and class attributes
    # First get class attributes (defaults defined at class level)
    config_dict = {k: v for k, v in config.__class__.__dict__.items() if not k.startswith('__') and not callable(v)}
    # Then update with instance attributes (those set/modified on the instance, which override class defaults)
    config_dict.update({k: v for k, v in vars(config).items() if not k.startswith('__')})

    np.save(filepath, config_dict, allow_pickle=True)

save_config(config, os.path.join(config.save_model_path, 'config.npy'))

set_seed(config.seed)

if config.model_type == 'snn':
    model = SNN(config).to(device)
else:
    model = SnnDelays(config).to(device)

if config.model_type == 'snn_delays_lr0':
    model.round_pos()

print(model)

print(f"===> Dataset    = {config.dataset}")
print(f"===> Model type = {config.model_type}")

if getattr(config, 'sparsity_p_delay', 0) > 0:
    print(f"===> Sparsity level for delay weights = {getattr(config, 'sparsity_p_delay', 0)}")
    print(f"===> Effective number of parameters = {utils.count_params_with_sparsity(model, config.sparsity_p_delay)}")
    wandb.log({"model_size": utils.count_params_with_sparsity(model, config.sparsity_p_delay)})
elif getattr(config, 'sparsity_p', 0) > 0:
    print(f"===> Sparsity level for weights = {getattr(config, 'sparsity_p', 0)}")
    print(f"===> Effective number of parameters = {utils.count_parameters(model)* (1 - config.sparsity_p)}") #TODO just roughly estimate
    wandb.log({"model_size": utils.count_parameters(model)* (1 - config.sparsity_p)})
else:
    print(f"===> Model size = {utils.count_parameters(model)}")
    wandb.log({"model_size": utils.count_parameters(model)})

print(f"===> Delay type = {config.delay_type}")
print(f"===> Hidden Layers = {config.n_hidden_layers}")
print(f"===> Max delay = {config.max_delay}")
print(f"===> Threshold = {config.v_threshold}")
if config.use_regularizers:
    print(f"===> Regularization factor = {config.reg_factor}")
    print(f"===> Regularization fmin = {config.reg_fmin}")
    print(f"===> Regularization fmax = {config.reg_fmax}")
#print(f"===> Time step = {config.time_step}")

if config.dataset == 'shd':
    train_loader, valid_loader = SHD_dataloaders(config)
    test_loader = None
elif config.dataset == 'shd_norm':
    train_loader, valid_loader, test_loader = SHD_NORM_dataloaders(config)
elif config.dataset == 'ssc':
    train_loader, valid_loader, test_loader = SSC_dataloaders(config)
elif config.dataset == 'gsc':
    train_loader, valid_loader, test_loader = GSC_dataloaders(config)
else:
    raise Exception(f'dataset {config.dataset} not implemented')

model.train_model(train_loader, valid_loader, test_loader, device)
