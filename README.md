# Axonal, Dendritic, and Synaptic Delays in SNNs

This repository contains source code to reproduce the experiments presented in the paper "Sparse Axonal and Dendritic Delays Enable Competitive SNNs for Keyword Classification" (2026), arXiv preprint [arXiv:2602.09746](https://arxiv.org/abs/2602.09746).
This implementation builds heavily on the original repository [https://github.com/Thvnvtos/SNN-delays](https://github.com/Thvnvtos/SNN-delays), which is the official implementation of "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings" (ICLR 2024). Compared to the original codebase, this version improves readability and overall structure, and includes the implementation of axonal and dendritic delays.

> **⚠️ Warning**: The code does not yet include all experiments from the paper, and some parts of the codebase have not been fully tested after recent cleanup and refactoring. These issues will be addressed very shortly with detailed experimentation.

## Installation

1. Install uv:

```bash
pip install uv

# or

pipx install uv
```

2. Clone the repository:

```bash
git clone git@github.com:YounesBouhadjar/AxDenSynDelaySNN.git
cd AxDenSynDelaySNN
```

3. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
uv pip install -e ".[dev]"
```

### SpikingJelly

Install SpikingJelly using:
```bash
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
git checkout 9844cdb65831f8076e18bd813d9e90bb53a59742
pip install -e .
cd ..
```

## Working with SNN-Delays

### Configuration

After installing all the dependencies is to specify the ```datasets_path``` in the config files.
Configuration files are located in the `configs/` directory. 

### Delay Types

This implementation supports three types of learnable delays:

- **Axonal delays**: Delays applied along the axon of neurons
- **Dendritic delays**: Delays applied along the dendrite of neurons  
- **Synaptic delays**: Delays applied at the synapse between neurons

The delay type can be specified in the configuration files using the `delay_type` parameter.
Accepted values are `axonal`, `dendritic`, and `synaptic`.

## Running Experiments

### Training a Model

To train a new model as defined by a config file for say axonal delays, simply use:
```bash
python main.py --config configs/SSC.yaml --delay_type axonal
```

and for sparse axonal delays:
```bash
python main.py --config configs/SSC.yaml --delay_type axonal --sparsity_p_delay 0.8
```

and for sparse axonal delays and sparse weights:
```bash
python main.py --config configs/SSC.yaml --delay_type axonal --sparsity_p_delay 0.8 --sparsity_p 0.6
```

If the ```use_wandb``` parameter is set to ```True```, the training and validation logs will be available at the wandb project specified in the configuration.

### Retrieve and Use W&B Sweeps

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and hyperparameter sweeps.

Hyperparameter sweep configuration files are located in the `wb_configs/` directory.

### Launch a Sweep

First, log in to W&B and execute:
```bash
python generate_sweep_id.py wb_configs/various_delays_gsc_all.yaml
```

The output will include a **Sweep ID** (e.g., `username/project/sweep_id`). Use this ID in the next step.

### Run the Agent

Now execute the agent to run the experiments:
```bash
wandb agent username/project/sweep_id
```

You can run multiple agents in parallel across machines to accelerate experimentation.

## Citation

If you use this code in your work, please cite the following paper:

* Bouhadjar, Y., Neftci, E. (2026). Sparse Axonal and Dendritic Delays Enable Competitive SNNs for Keyword Classification. [arXiv:2602.09746](https://arxiv.org/abs/2602.09746)

```bibtex
@misc{Bouhadjar26_sparsedelays,
      title={Sparse Axonal and Dendritic Delays Enable Competitive SNNs for Keyword Classification}, 
      author={Younes Bouhadjar and Emre Neftci},
      year={2026},
      eprint={2602.09746},
      publisher={arXiv},
      doi={10.48550/arXiv.2602.09746},
      url={https://arxiv.org/abs/2602.09746}, 
}
```

## Project Structure

```
refSNN-delays/
├── configs/          # Configuration files for different datasets and experiments
├── wb_configs/       # W&B sweep config files for hyperparameter searches
├── models/           # Model implementations (SNN with delays)
├── datasets.py       # Dataset loading and preprocessing
├── train.py          # Training script
├── main.py           # Main entry point
└── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Note:** This code builds heavily on the original implementation from [https://github.com/Thvnvtos/SNN-delays](https://github.com/Thvnvtos/SNN-delays). The original repository is the official implementation of "Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings" [https://openreview.net/forum?id=4r2ybzJnmN](https://openreview.net/forum?id=4r2ybzJnmN) (ICLR 2024).
