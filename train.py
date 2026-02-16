# SPDX-License-Identifier: MIT
# Copyright (c) 2026-present
"""
Training module for SNN models with delays.

This code is modified from:
https://github.com/Thvnvtos/SNN-delays
"""

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import set_seed
from datasets import Augs
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4
import os

# Import neuron for SiLIFNode parameter collection
try:
    from spikingjelly.activation_based import neuron
except ImportError:
    neuron = None

eventid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.build_model()
        self.init_model()

        self.init_pos = []
        if self.config.model_type != 'snn':
            # Extract positions from delay layers in snn structure
            if hasattr(self, 'snn'):
                for layer in self.snn:
                    if hasattr(layer, 'delay_layer') and hasattr(layer.delay_layer, 'P'):
                        self.init_pos.append(np.copy(layer.delay_layer.P.cpu().detach().numpy()))
    
    def _get_delay_layer(self, i):
        """Helper method to get delay layer at index i."""
        if hasattr(self, 'snn') and i < len(self.snn):
            layer = self.snn[i]
            if hasattr(layer, 'delay_layer'):
                return layer.delay_layer
        return None
    
    def _get_batchnorm_layer(self, i):
        """Helper method to get BatchNorm layer at index i."""
        if hasattr(self, 'snn') and i < len(self.snn):
            layer = self.snn[i]
            if hasattr(layer, 'norm') and layer.normalize:
                return layer.norm
        return None


    def optimizers(self):
        ##################################
        #  returns a list of optimizers
        ##################################
        optimizers_return = []

        if self.config.model_type in ['snn_delays', 'snn_delays_lr0', 'snn']:

            if self.config.optimizer_w == 'adam':
                optimizers_return.append(optim.Adam([{'params':self.weights, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                                     {'params':self.weights_plif, 'lr':self.config.lr_w, 'weight_decay':self.config.weight_decay},
                                                     {'params':self.weights_bn, 'lr':self.config.lr_w, 'weight_decay':0}]))
            if self.config.model_type == 'snn_delays':
                if self.config.optimizer_pos == 'adam':
                    optimizers_return.append(optim.Adam(self.positions, lr = self.config.lr_pos, weight_decay=0))
        elif self.config.model_type == 'ann':
            if self.config.optimizer_w == 'adam':
                optimizers_return.append(optim.Adam(self.model.parameters(), lr = self.config.lr_w, betas=(0.9,0.999)))

        return optimizers_return


    def schedulers(self, optimizers):
        ##################################
        #  returns a list of schedulers
        #  if self.config.scheduler_x is none:  list will be empty
        ##################################
        schedulers_return = []

        if self.config.model_type in ['snn_delays', 'snn_delays_lr0','snn']:
            if self.config.scheduler_w == 'one_cycle':
                schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w,
                                                                             total_steps=self.config.epochs))
            elif self.config.scheduler_w == 'cosine_a':
                schedulers_return.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0],
                                                                                        T_max = self.config.t_max_w))

            if self.config.model_type == 'snn_delays':
                if self.config.scheduler_pos == 'one_cycle':
                    schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[1], max_lr=self.config.max_lr_pos,
                                                                                total_steps=self.config.epochs))
                elif self.config.scheduler_pos == 'cosine_a':
                    schedulers_return.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1],
                                                                                        T_max = self.config.t_max_pos))

        elif self.config.model_type == 'ann':
            if self.config.scheduler_w == 'one_cycle':
                schedulers_return.append(torch.optim.lr_scheduler.OneCycleLR(optimizers[0], max_lr=self.config.max_lr_w,
                                                                             total_steps=self.config.epochs))

        return schedulers_return


    def fused_bias(self, conv, bn):
        """Compute fused bias from conv+bn layers."""
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_mean)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        std = torch.sqrt(var + eps)

        fused_bias = beta + (b - mean) * (gamma / std)
        return fused_bias

    def negative_fused_bias_penalty(self, conv, bn, weight=1e-3):
        """Penalize positive fused bias values from conv+bn layers."""
        fused_bias = self.fused_bias(conv, bn)
        penalty = torch.relu(fused_bias).sum()  # penalize positive ones
        return weight * penalty


    def calc_loss(self, output, y):

        if self.config.loss == 'mean': m = torch.mean(output, 0)
        elif self.config.loss == 'max': m, _ = torch.max(output, 0)
        elif self.config.loss == 'spike_count': m = torch.sum(output, 0)
        elif self.config.loss == 'sum':
            softmax_fn = nn.Softmax(dim=2)
            m = torch.sum(softmax_fn(output), 0)

        # probably better to add it in init, or in general do it one time only
        if self.config.loss_fn == 'CEloss':
            #compare using this to directly using nn.CrossEntropyLoss

            CEloss = nn.CrossEntropyLoss()
            loss = CEloss(m, y)
            #log_softmax_fn = nn.LogSoftmax(dim=1)
            #loss_fn = nn.NLLLoss()
            #log_p_y = log_softmax_fn(m)
            #loss = loss_fn(log_p_y, y)

            return loss


    def calc_metric(self, output, y):
        # mean accuracy over batch
        if self.config.loss == 'mean': m = torch.mean(output, 0)
        elif self.config.loss == 'max': m, _ = torch.max(output, 0)
        elif self.config.loss == 'spike_count': m = torch.sum(output, 0)
        elif self.config.loss == 'sum':
            softmax_fn = nn.Softmax(dim=2)
            m = torch.sum(softmax_fn(output), 0)

        return np.mean((torch.max(y,1)[1]==torch.max(m,1)[1]).detach().cpu().numpy())


    def fine_tune(self, train_loader, valid_loader, test_loader, device):

        #if self.config.spiking_neuron_type == 'plif' and self.config.spiking_neuron_type_finetuning == 'lif':

        self.config.DCLSversion = 'max'
        self.config.model_type = 'snn_delays_lr0'

        self.config.lr_w = self.config.lr_w_finetuning
        self.config.max_lr_w = self.config.max_lr_w_finetuning

        self.config.dropout_p = self.config.dropout_p_finetuning
        self.config.stateful_synapse_learnable = self.config.stateful_synapse_learnable_finetuning
        self.config.spiking_neuron_type = self.config.spiking_neuron_type_finetuning
        self.config.epochs = self.config.epochs_finetuning

        self.config.final_epoch = 0

        self.config.wandb_run_name = self.config.wandb_run_name_finetuning
        self.config.wandb_group_name = self.config.wandb_group_name_finetuning


        self.__init__(self.config)
        self.to(device)
        model_path = os.path.join(self.config.save_model_path, 'model.pt')
        self.load_state_dict(torch.load(model_path), strict=False)

        if hasattr(self, 'snn'):
            for i, layer in enumerate(self.snn):
                delay_layer = self._get_delay_layer(i)
                if delay_layer is not None and hasattr(delay_layer, 'SIG'):
                    delay_layer.SIG *= 0

        if hasattr(self, 'round_pos'):
            self.round_pos()

        # Update save path for fine-tuning (create new directory inside ckpt_models)
        self.config.save_model_path = os.path.join('ckpt_models', self.config.save_model_path_finetuning)
        os.makedirs(self.config.save_model_path, exist_ok=True)
        self.train_model(train_loader, valid_loader, test_loader, device)



    def train_model(self, train_loader, valid_loader, test_loader, device):

        #######################################################################################
        #           Main Training Loop for all models
        #
        #
        #
        ##################################    Initializations    #############################

        #set_seed(self.config.seed)

        if getattr(self.config, 'test_pretrained_model', False):
            fname = os.path.join(self.config.save_model_path, 'model.pt')
            print(f"Loading best ACC model:", fname)
            self.load_state_dict(torch.load(fname))

            loss_valid, metric_valid, avg_spikes_valid, ops_valid, firing_rates_valid, fused_bias_penalty_valid, voltage_reg_valid, max_population_frs_valid = self.eval_model(valid_loader, device)
            print(f"Loss Valid = {loss_valid:.3f}  |  Acc Valid = {100*metric_valid:.2f}%  |  Avg Spikes Valid = {avg_spikes_valid:.3f}  |  Ops Valid = {ops_valid:.3f}")
            wandb.log({"loss_valid":loss_valid,
                       "acc_valid":100*metric_valid,
                       "avg_spikes_valid":avg_spikes_valid,
                       "ops_valid":ops_valid,
                       "fused_bias_penalty_valid":fused_bias_penalty_valid,
                       "voltage_reg_valid":voltage_reg_valid,
                       "max_population_frs_valid":max_population_frs_valid})

            if test_loader:
                loss_test, metric_test, avg_spikes_test, ops_test, firing_rates_test, fused_bias_penalty_test, voltage_reg_test, max_population_frs_test = self.eval_model(test_loader, device)
            else:
                # could be improved
                loss_test, metric_test, avg_spikes_test, ops_test, firing_rates_test, fused_bias_penalty_test, voltage_reg_test, max_population_frs_test = 100, 0, 0, 0, 0, 0, 0, 0

            print(f"Loss Test = {loss_test:.3f}  |  Acc Test = {100*metric_test:.2f}%  |  Avg Spikes Test = {avg_spikes_test:.3f}  |  Ops Test = {ops_test:.3f}")
            wandb.log({"loss_test":loss_test,
                       "acc_test":100*metric_test,
                       "avg_spikes_test":avg_spikes_test,
                       "ops_test":ops_test,
                       "fused_bias_penalty_test":fused_bias_penalty_test,
                       "voltage_reg_test":voltage_reg_test,
                       "max_population_frs_test":max_population_frs_test})
            
            return
 
        optimizers = self.optimizers()
        schedulers = self.schedulers(optimizers)

        augmentations = Augs(self.config)

        ##################################    Train Loop    ##############################


        loss_epochs = {'train':[], 'valid':[] , 'test':[]}
        metric_epochs = {'train':[], 'valid':[], 'test':[]}
        avg_spikes_epochs = {'train':[], 'valid':[], 'test':[]}
        ops_epochs = {'train':[], 'valid':[], 'test':[]}
        firing_rates_epochs = {'train':[], 'valid':[], 'test':[]}
        fused_bias_penalty_epochs = {'train':[], 'valid':[], 'test':[]}
        voltage_reg_epochs = {'train':[], 'valid':[], 'test':[]}
        max_population_frs_epochs = {'train':[], 'valid':[], 'test':[]}
        best_metric_val = 0 #1e6
        best_avg_spikes_val = 0 #1e6
        best_avg_spikes_test = 0 #1e6
        best_ops_val = 0 #1e6
        best_ops_test = 0 #1e6
        best_metric_test = 0 #1e6
        best_loss_val = 1e6

        pre_pos_epoch = self.init_pos.copy()
        pre_pos_5epochs = self.init_pos.copy()
        batch_count = 0
        for epoch in range(self.config.epochs):
            self.train()
            #last element in the tuple corresponds to the collate_fn return
            loss_batch, metric_batch, avg_spikes_batch, ops_batch, firing_rates_batch, fused_bias_penalty_batch, voltage_reg_batch, max_population_frs_batch = [], [], [], [], [], [], [], []
            pre_pos = pre_pos_epoch.copy()
            for i, (x, y, _) in enumerate(tqdm(train_loader, mininterval=1)):
                # x for shd and ssc is: (batch, time, neurons)

                y = F.one_hot(y, self.config.n_outputs).float()

                x = x.permute(1,0,2).float().to(device)  #(time, batch, neurons)
                y = y.to(device)

                for opt in optimizers: opt.zero_grad()

                output, avg_spikes_ops = self.forward(x)
                avg_spikes, firing_rates, ops, voltage_reg_total, max_population_frs = avg_spikes_ops
                loss = self.calc_loss(output, y)

                if getattr(self.config, 'use_regularizers', False):
                    reg_quiet = F.relu(self.config.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.config.reg_fmax).sum()
                    loss += self.config.reg_factor * (reg_quiet + reg_burst)
                
                if getattr(self.config, 'use_voltage_reg', False):
                    loss += self.config.voltage_reg_factor * voltage_reg_total

                # Add negative fused bias penalty for conv/linear + batchnorm layers
                batch_penalty = 0.0
                if self.config.use_batchnorm and getattr(self.config, 'conv_bn_penalty', False):
                    penalty_weight = getattr(self.config, 'bn_penalty_weight', 1e-3)
                    delay_type = getattr(self.config, 'delay_type', 'synaptic')
                    
                    # For synaptic delays: Dcls1d + BatchNorm are in snn layers
                    # For axonal/dendritic delays: separate Linear layers (W, Wh, Wf) + BatchNorm
                    if delay_type == 'synaptic':
                        # Synaptic delays: layer.delay_layer is Dcls1d, layer.norm is BatchNorm
                        if hasattr(self, 'snn'):
                            for i, layer in enumerate(self.snn):
                                delay_layer = self._get_delay_layer(i)
                                bn_layer = self._get_batchnorm_layer(i)
                                if delay_layer is not None and bn_layer is not None:
                                    penalty = self.negative_fused_bias_penalty(delay_layer, bn_layer, weight=penalty_weight)
                                    loss += penalty
                                    batch_penalty += penalty.detach().cpu().item()
                    else:
                        # Axonal/dendritic delays: Linear layers (W, Wh, Wf) are in layer attributes
                        # BatchNorm is in layer.norm and should be fused with the corresponding Linear layer
                        if hasattr(self, 'snn'):
                            for i, layer in enumerate(self.snn):
                                if i == len(self.snn) - 1:  # Skip output layer
                                    continue
                                bn_layer = self._get_batchnorm_layer(i)
                                if bn_layer is not None:
                                    # Get corresponding linear layer
                                    if i == 0 and hasattr(layer, 'W'):
                                        linear_layer = layer.W
                                    elif hasattr(layer, 'Wh'):
                                        linear_layer = layer.Wh
                                    else:
                                        continue
                                    penalty = self.negative_fused_bias_penalty(linear_layer, bn_layer, weight=penalty_weight)
                                    loss += penalty
                                    batch_penalty += penalty.detach().cpu().item()

                loss.backward()
                for opt in optimizers: opt.step()

                # Clamp weights to specified range if enabled
                if getattr(self.config, 'clamp_weights', False):
                    weight_min = getattr(self.config, 'weight_min', -2.0)
                    weight_max = getattr(self.config, 'weight_max', 2.0)
                    
                    # Clamp DCLS weights
                    if hasattr(self, 'snn'):
                        for layer in self.snn:
                            delay_layer = layer.delay_layer if hasattr(layer, 'delay_layer') else None
                            if delay_layer is not None and hasattr(delay_layer, 'weight'):
                                with torch.no_grad():
                                    delay_layer.weight.clamp_(weight_min, weight_max)
                    
                    # Clamp linear layer weights for axonal/dendritic delays
                    delay_type = getattr(self.config, 'delay_type', 'synaptic')
                    if delay_type in ['axonal', 'dendritic']:
                        if hasattr(self, 'W') and hasattr(self.W, 'weight'):
                            with torch.no_grad():
                                self.W.weight.clamp_(weight_min, weight_max)
                        if hasattr(self, 'Wh'):
                            try:
                                for w in self.Wh:
                                    if hasattr(w, 'weight'):
                                        with torch.no_grad():
                                            w.weight.clamp_(weight_min, weight_max)
                            except TypeError:
                                if hasattr(self.Wh, 'weight'):
                                    with torch.no_grad():
                                        self.Wh.weight.clamp_(weight_min, weight_max)
                        if hasattr(self, 'Wf') and hasattr(self.Wf, 'weight'):
                            with torch.no_grad():
                                self.Wf.weight.clamp_(weight_min, weight_max)

                metric = self.calc_metric(output, y)

                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)
                avg_spikes_batch.append(avg_spikes.sum())
                ops_batch.append(ops.sum())
                firing_rates_batch.append(firing_rates.mean().detach().cpu().item())
                fused_bias_penalty_batch.append(batch_penalty)
                voltage_reg_batch.append(voltage_reg_total.detach().cpu().item())
                max_population_frs_batch.append(max_population_frs.mean().detach().cpu().item())

                self.reset_model(train=True)

                if self.config.use_wandb and getattr(self.config, 'debug', False) and self.config.model_type == 'snn_delays':
                    wandb_pos_log = {}
                    if hasattr(self, 'snn'):
                        for b, layer in enumerate(self.snn):
                            delay_layer = self._get_delay_layer(b)
                            if delay_layer is not None and hasattr(delay_layer, 'P') and b < len(pre_pos):
                                curr_pos = delay_layer.P.cpu().detach().numpy()
                                wandb_pos_log[f'dpos{b}'] = np.abs(curr_pos - pre_pos[b]).mean()
                                pre_pos[b] = curr_pos.copy()
                    
                    wandb_pos_log.update({"batch":batch_count})
                    wandb.log(wandb_pos_log)
                    batch_count += 1


            if self.config.model_type == 'snn_delays':
                pos_logs = {}
                if hasattr(self, 'snn'):
                    for b in range(len(self.snn)):
                        if b < len(pre_pos) and b < len(pre_pos_epoch):
                            pos_logs[f'dpos{b}_epoch'] = np.abs(pre_pos[b] - pre_pos_epoch[b]).mean()
                            pre_pos_epoch[b] = pre_pos[b].copy()

                if epoch%5==0 and epoch>0:
                    if hasattr(self, 'snn'):
                        for b in range(len(self.snn)):
                            if b < len(pre_pos) and b < len(pre_pos_5epochs):
                                pos_logs[f'dpos{b}_5epochs'] = np.abs(pre_pos[b] - pre_pos_5epochs[b]).mean()
                                pre_pos_5epochs[b] = pre_pos[b].copy()


            loss_epochs['train'].append(np.mean(loss_batch))
            metric_epochs['train'].append(np.mean(metric_batch))
            avg_spikes_epochs['train'].append(np.mean(avg_spikes_batch))
            ops_epochs['train'].append(np.mean(ops_batch))
            firing_rates_epochs['train'].append(np.mean(firing_rates_batch))
            voltage_reg_epochs['train'].append(np.mean(voltage_reg_batch))
            fused_bias_penalty_epochs['train'].append(np.mean(fused_bias_penalty_batch))
            max_population_frs_epochs['train'].append(np.mean(max_population_frs_batch))

            for scheduler in schedulers: scheduler.step()
            self.decrease_sig(epoch)
            if getattr(self.config, 'sparsity_p_delay', 0) > 0:
                self.decrease_sparsity(epoch)

            ##################################    Eval Loop    #########################


            loss_valid, metric_valid, avg_spikes_valid, ops_valid, firing_rates_valid, fused_bias_penalty_valid, voltage_reg_valid, max_population_frs_valid = self.eval_model(valid_loader, device)

            loss_epochs['valid'].append(loss_valid)
            metric_epochs['valid'].append(metric_valid)
            avg_spikes_epochs['valid'].append(avg_spikes_valid)
            ops_epochs['valid'].append(ops_valid)
            firing_rates_epochs['valid'].append(firing_rates_valid)
            fused_bias_penalty_epochs['valid'].append(fused_bias_penalty_valid)
            voltage_reg_epochs['valid'].append(voltage_reg_valid)
            max_population_frs_epochs['valid'].append(max_population_frs_valid)

            if test_loader:
                loss_test, metric_test, avg_spikes_test, ops_test, firing_rates_test, fused_bias_penalty_test, voltage_reg_test, max_population_frs_test = self.eval_model(test_loader, device)
            else:
                # could be improved
                loss_test, metric_test, avg_spikes_test, ops_test, firing_rates_test, fused_bias_penalty_test, voltage_reg_test, max_population_frs_test = 100, 0, 0, 0, 0, 0, 0, 0

            loss_epochs['test'].append(loss_test)
            metric_epochs['test'].append(metric_test)
            avg_spikes_epochs['test'].append(avg_spikes_test)
            ops_epochs['test'].append(ops_test)
            firing_rates_epochs['test'].append(firing_rates_test)
            fused_bias_penalty_epochs['test'].append(fused_bias_penalty_test)
            voltage_reg_epochs['test'].append(voltage_reg_test)
            max_population_frs_epochs['test'].append(max_population_frs_test)

            if  metric_valid > best_metric_val:#  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Metric model...")
                if metric_valid > 0.7:
                    model_path = os.path.join(self.config.save_model_path, 'model.pt')
                    torch.save(self.state_dict(), model_path)
                best_metric_val = metric_valid
                best_avg_spikes_val = avg_spikes_valid
                best_ops_val = ops_valid

                # TODO: Rethink naming. `best_metric_test` is misleading; it suggests the best test score across models.
                best_metric_test = metric_test
                best_avg_spikes_test = avg_spikes_test
                best_ops_test = ops_test
 
            if  loss_valid < best_loss_val:#  and (self.config.model_type != 'snn_delays' or epoch >= self.config.final_epoch - 1):
                print("# Saving best Loss model...")
                model_path = os.path.join(self.config.save_model_path, 'best_loss.pt')
                torch.save(self.state_dict(), model_path)
                best_loss_val = loss_valid
            
            ########################## Logging and Plotting  ##########################

            print(f"=====> Epoch {epoch} : \nLoss Train = {loss_epochs['train'][-1]:.3f}  |  Acc Train = {100*metric_epochs['train'][-1]:.2f}% | Firing Rates Train = {firing_rates_epochs['train'][-1]:.3f}")
            print(f"Loss Valid = {loss_epochs['valid'][-1]:.3f}  |  Acc Valid = {100*metric_epochs['valid'][-1]:.2f}%  |  Best Acc Valid = {100*max(metric_epochs['valid'][-1], best_metric_val):.2f}%  |  Avg Spikes Valid = {avg_spikes_epochs['valid'][-1]:.3f}" )

            if test_loader:
                print(f"Loss Test = {loss_epochs['test'][-1]:.3f}  |  Acc Test = {100*metric_epochs['test'][-1]:.2f}%  |  Best Acc Test = {100*max(metric_epochs['test'][-1], best_metric_test):.2f}%  |  Avg Spikes Test = {avg_spikes_epochs['test'][-1]:.3f}")

            if self.config.use_wandb:

                if self.config.scheduler_w != 'none':
                    lr_w = schedulers[0].get_last_lr()[0]
                    k_pos = 1
                else:
                    lr_w = self.config.lr_w
                    k_pos = 0

                lr_pos = schedulers[k_pos].get_last_lr()[0] if self.config.model_type == 'snn_delays' and self.config.scheduler_pos != 'none' else self.config.lr_pos

                wandb_logs = {"Epoch":epoch,
                              "loss_train":loss_epochs['train'][-1],
                              "acc_train" : metric_epochs['train'][-1],
                              "avg_spikes_train" : avg_spikes_epochs['train'][-1],
                              "ops_train" : ops_epochs['train'][-1],
                              "firing_rates_train" : firing_rates_epochs['train'][-1],
                              "fused_bias_penalty_train" : fused_bias_penalty_epochs['train'][-1],
                              "voltage_reg_train" : voltage_reg_epochs['train'][-1],
                              "max_population_frs_train" : max_population_frs_epochs['train'][-1],
                              "loss_valid" : loss_epochs['valid'][-1],
                              "acc_valid" : metric_epochs['valid'][-1],
                              "best_acc_valid" : max(metric_epochs['valid'][-1], best_metric_val),
                              "best_avg_spikes_valid" : best_avg_spikes_val,
                              "best_ops_valid" : best_ops_val,
                              "avg_spikes_valid" : avg_spikes_epochs['valid'][-1],
                              "ops_valid" : ops_epochs['valid'][-1],
                              "firing_rates_valid" : firing_rates_epochs['valid'][-1],
                              "fused_bias_penalty_valid" : fused_bias_penalty_epochs['valid'][-1],
                              "voltage_reg_valid" : voltage_reg_epochs['valid'][-1],
                              "max_population_frs_valid" : max_population_frs_epochs['valid'][-1],
                              "loss_test" : loss_epochs['test'][-1],
                              "acc_test" : metric_epochs['test'][-1],
                              "best_acc_test" : max(metric_epochs['test'][-1], best_metric_test),
                              "best_avg_spikes_test" : best_avg_spikes_test,
                              "best_ops_test" : best_ops_test,
                              "avg_spikes_test" : avg_spikes_epochs['test'][-1],
                              "ops_test" : ops_epochs['test'][-1],
                              "firing_rates_test" : firing_rates_epochs['test'][-1],
                              "fused_bias_penalty_test" : fused_bias_penalty_epochs['test'][-1],
                              "voltage_reg_test" : voltage_reg_epochs['test'][-1],
                              "max_population_frs_test" : max_population_frs_epochs['test'][-1],
                              "lr_w" : lr_w,
                              "lr_pos" : lr_pos}

                if getattr(self.config, 'sparsity_p_delay', 0) > 0:
                    if hasattr(self, 'snn') and len(self.snn) > 0:
                        delay_layer = self._get_delay_layer(0)
                        if delay_layer is not None and hasattr(delay_layer, 'scheduled_sparsity'):
                            print("scheduled_sparsity: ", delay_layer.scheduled_sparsity())
                            wandb_logs.update({"scheduled_sparsity": delay_layer.scheduled_sparsity()})

                if getattr(self.config, 'debug', False):
                    model_logs = self.get_model_wandb_logs()

                    wandb_logs.update(model_logs)

                    if self.config.model_type == 'snn_delays':
                        wandb_logs.update(pos_logs)

                wandb.log(wandb_logs)

        if self.config.use_wandb:
            wandb.run.finish()

    def eval_model(self, loader, device):
        # Save model state in memory (not to file)
        saved_state = self.state_dict()
        self.eval()
        with torch.no_grad():

            if self.config.model_type != 'snn':
                for i, layer in enumerate(self.snn):
                    delay_layer = self._get_delay_layer(i)
                    if delay_layer is not None:
                        if hasattr(delay_layer, 'SIG'):
                            delay_layer.SIG *= 0
                        if hasattr(delay_layer, 'version'):
                            delay_layer.version = 'max'
                        if hasattr(delay_layer, 'DCK') and hasattr(delay_layer.DCK, 'version'):
                            delay_layer.DCK.version = 'max'

                if hasattr(self, 'round_pos'):
                    self.round_pos()

            # Adjust BN bias to push all fused bias ≤ -margin
            if self.config.use_batchnorm and getattr(self.config, 'conv_bn_penalty', False):
                delay_type = getattr(self.config, 'delay_type', 'synaptic')
                
                if delay_type == 'axonal' or delay_type == 'dendritic':
                    for i, layer in enumerate(self.snn):
                        if i == len(self.snn) - 1:  # Skip output layer
                            continue
                        bn_layer = self._get_batchnorm_layer(i)
                        if bn_layer is not None and bn_layer.bias is not None:
                            fb = self.fused_bias(layer.W, bn_layer)
                            bn_layer.bias.sub_(torch.relu(fb) + 1e-3)  # push all b' ≤ -margin
                else:
                    raise ValueError(f"Delay type: {delay_type} not supported for BN bias adjustment")


            loss_batch, metric_batch, avg_spikes_batch, ops_batch, firing_rates_batch, fused_bias_penalty_batch, voltage_reg_batch, max_population_frs_batch = [], [], [], [], [], [], [], []
            for i, (x, y, _) in enumerate(tqdm(loader)):

                y = F.one_hot(y, self.config.n_outputs).float()

                x = x.permute(1,0,2).float().to(device)
                y = y.to(device)

                output, avg_spikes_ops = self.forward(x)
                avg_spikes, firing_rates, ops, voltage_reg_total, max_population_frs = avg_spikes_ops

                loss = self.calc_loss(output, y)
                metric = self.calc_metric(output, y)
                 
                # Compute fused bias penalty (same logic as training)
                batch_penalty = 0.0
                if self.config.use_batchnorm and getattr(self.config, 'conv_bn_penalty', False):
                    penalty_weight = getattr(self.config, 'bn_penalty_weight', 1e-3)
                    delay_type = getattr(self.config, 'delay_type', 'synaptic')
                    
                    if delay_type == 'axonal' or delay_type == 'dendritic':

                        for i, layer in enumerate(self.snn):
                            if i == len(self.snn) - 1:  # Skip output layer
                                continue
                            bn_layer = self._get_batchnorm_layer(i)
                            if bn_layer is not None:
                                if i == 0 and hasattr(layer, 'W'):
                                    penalty = self.negative_fused_bias_penalty(layer.W, bn_layer, weight=penalty_weight)
                                    batch_penalty += penalty.detach().cpu().item()
                                elif hasattr(layer, 'Wh'):
                                    penalty = self.negative_fused_bias_penalty(layer.Wh, bn_layer, weight=penalty_weight)
                                    batch_penalty += penalty.detach().cpu().item()
                    else:
                        raise ValueError(f"Delay type: {delay_type} not supported for BN bias adjustment")

                avg_spikes_batch.append(avg_spikes.sum())
                ops_batch.append(ops.sum())
                loss_batch.append(loss.detach().cpu().item())
                metric_batch.append(metric)
                firing_rates_batch.append(firing_rates.mean().detach().cpu().item())
                fused_bias_penalty_batch.append(batch_penalty)
                voltage_reg_batch.append(voltage_reg_total.detach().cpu().item())
                max_population_frs_batch.append(max_population_frs.mean().detach().cpu().item())

                self.reset_model(train=False)

            if self.config.DCLSversion == 'gauss' and self.config.model_type != 'snn':
                for i, layer in enumerate(self.snn):
                    delay_layer = self._get_delay_layer(i)
                    if delay_layer is not None:
                        if hasattr(delay_layer, 'version'):
                            delay_layer.version = 'gauss'
                        if hasattr(delay_layer, 'DCK') and hasattr(delay_layer.DCK, 'version'):
                            delay_layer.DCK.version = 'gauss'

            # Log weights per synaptic delay type (distributions)
            delay_type = getattr(self.config, 'delay_type', 'synaptic')
            if self.config.use_wandb and getattr(self.config, 'debug', False):
                weight_logs = {}

                def _hist(prefix, tensor):
                    if tensor is None:
                        return
                    
                    t = tensor.detach().float().cpu().view(-1).numpy()
                    if t.size == 0:
                        return

                    weight_logs[f'{prefix}_hist'] = wandb.Histogram(t, num_bins=64)

                # Log weights of first layer
                if delay_type == 'synaptic':
                    if len(self.snn) > 0:
                        delay_layer = self._get_delay_layer(0)
                        if delay_layer is not None and hasattr(delay_layer, 'weight'):
                            w = delay_layer.weight
                            _hist(f'dcls_w_{0}', w)

                elif delay_type == 'axonal':
                    w = self.W.weight
                    _hist(f'dcls_w_{0}', w)

                elif delay_type == 'dendritic':
                    w = self.W.weight
                    _hist(f'dcls_w_{0}', w)

                else:
                    raise ValueError(f"Invalid delay type: {delay_type}")

                if len(weight_logs) > 0:
                    wandb.log(weight_logs)
            
            # Restore original model state from memory
            self.load_state_dict(saved_state, strict=True)

        return np.mean(loss_batch), np.mean(metric_batch), np.mean(avg_spikes_batch), np.mean(ops_batch), np.mean(firing_rates_batch), np.mean(fused_bias_penalty_batch), np.mean(voltage_reg_batch), np.mean(max_population_frs_batch)
