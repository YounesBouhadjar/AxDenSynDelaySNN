import os
import sys
import yaml
import wandb

# get commmand line arguments
try:
    config_wb = sys.argv[1]
except:
    print("provide weight and biases config file!, see folder: wb_configs")
    exit(-1)

with open(config_wb, 'r') as cfgfile:
    params_wb = yaml.load(cfgfile, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(params_wb)

print("sweep_id:", sweep_id)

# Note: run the following command directly in the command line or in a different script
# just to avoid running the same parameter space using different sweep_id
# os.system('wandb agent --count 500 %s\n' % (sweep_id))
