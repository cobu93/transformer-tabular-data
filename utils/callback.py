import skorch
import wandb
import torch
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import tempfile
import os
import re

matplotlib.use("agg")

    
class ReportWandB(skorch.callbacks.Callback):    
    
    def __init__(self, *args, **kwargs):
        super(ReportWandB, self).__init__(*args, **kwargs)

        self.metrics = {}

    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        
        excluded = ["batches"]

        last_history = net.history[-1]
        
        for key in last_history.keys():

            if key in excluded:
                continue

            if "_best" in key:
                if last_history[key]:
                    self.metrics[key.replace("_best", "_opt")] = last_history[key.replace("_best", "")]
            else:
                self.metrics[key] = last_history[key]
        
        logvars = net.module_.categorical_embedding.categories_logvars.detach().cpu().numpy()
        means = net.module_.categorical_embedding.categories_means

        if means.shape[0] > 0:
            norm_means = (means.T / torch.norm(means, dim=-1)).T.detach().cpu().numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:
                generate_matrix_image(logvars, os.path.join(tmpdirname, "logvars.png"))
                generate_matrix_image(norm_means, os.path.join(tmpdirname, "norm_means.png"))
                
                wandb.log({
                    **self.metrics,
                    "logvars": wandb.Image(os.path.join(tmpdirname, "logvars.png")),
                    "norm_means": wandb.Image(os.path.join(tmpdirname, "norm_means.png"))
                    })
        else:
            wandb.log(self.metrics)

def generate_matrix_image(matrix, name):
    ax = sns.heatmap(matrix)  
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.set_ylabel("")

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.set_xlabel("")

    ax.figure.savefig(name, bbox_inches="tight", pad_inches=0)
    plt.clf()

class LoadPartialInit(skorch.callbacks.Callback):    
    def __init__(self, checkpoint, restart_keys_regex=[]):
        self.checkpoint = checkpoint
        self.restart_keys_regex = restart_keys_regex

    def initialize(self):
        self.did_load_ = False
        return self

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        if not self.did_load_:
            self.did_load_ = True

            def _get_state_dict(f_name):
                map_location = skorch.utils.get_map_location(net.device)
                net.device = net._check_device(net.device, map_location)
                return torch.load(f_name, map_location=map_location)
            
            kwargs_full = {}
            
            if not net.initialized_:
                net.initialize()

            if self.checkpoint.f_history is not None:
                net.history = skorch.history.History.from_file(self.checkpoint.f_history_)
            
            kwargs_full.update(**self.checkpoint.get_formatted_files(net))
            net.history.clear()
            
            kwargs_module, kwargs_other = skorch.utils._check_f_arguments("load_params", **kwargs_full)

            if not kwargs_module and not kwargs_other:
                print("Nothing to load")
                return
            
            
            msg_init = (
                "Cannot load state of an un-initialized model. "
                "Please initialize first by calling .initialize() "
                "or by fitting the model with .fit(...).")
            msg_module = (
                "You are trying to load 'f_{name}' but for that to work, the net "
                "needs to have an attribute called 'net.{name}_' that is a PyTorch "
                "Module or Optimizer; make sure that it exists and check for typos.")

            for attr, f_name in kwargs_module.items():

                if attr not in ["module_"]:
                    continue


                # valid attrs can be 'module_', 'optimizer_', etc.
                if attr.endswith("_") and not net.initialized_:
                    net.check_is_fitted([attr], msg=msg_init)
                
                module = net._get_module(attr, msg=msg_module)
                loaded_state_dict = _get_state_dict(f_name)
                transfer_dict = {}

                current_state_dict = module.state_dict()
                
                for k, v in loaded_state_dict.items():

                    if k not in current_state_dict:
                        print(f"Ignoring {k}")
                        continue
                    
                    transfer_dict[k] = v

                    for rs_key in self.restart_keys_regex:
                        if re.search(rs_key, k):
                            new_shape = current_state_dict[k].shape
                            old_tensor = transfer_dict[k]
                            device = old_tensor.device

                            print(f"Restarting {k}. {old_tensor.shape} -> {new_shape}")
                            old_mean = torch.mean(old_tensor)
                            old_std = torch.std(old_tensor)
                            #new_tensor = torch.randn(*new_shape).to(device) * old_std + old_mean
                            new_tensor = old_tensor[:new_shape[0]]
                            transfer_dict[k] = new_tensor
                            break

                for k, v in current_state_dict.items():
                    if k not in transfer_dict:
                        print(f"Adding {k}.")
                        transfer_dict[k] = v
                
                #state_dict.update(pretrained_dict)
                module.load_state_dict(transfer_dict)
                
def get_default_callbacks(early_stopping_params={}, include_report_wandb=True):

    
    callbacks = [
        ("early_stopping", skorch.callbacks.EarlyStopping(**early_stopping_params)),
        ("duration", skorch.callbacks.EpochTimer())
        ]

    if include_report_wandb:
        callbacks.append(
            ("report_wandb", ReportWandB())
        )

    return callbacks

def get_classification_callbacks(early_stopping_params={}, include_report_wandb=True, multiclass=True):

    callbacks = [
        ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
        ("accuracy", skorch.callbacks.EpochScoring("accuracy", lower_is_better=False)),
        ]

    if not multiclass:
        callbacks.extend([
            ("roc_auc", skorch.callbacks.EpochScoring("roc_auc", lower_is_better=False)),
            ("f1", skorch.callbacks.EpochScoring("f1", lower_is_better=False)),
            ("precision", skorch.callbacks.EpochScoring("precision", lower_is_better=False)),
            ("recall", skorch.callbacks.EpochScoring("recall", lower_is_better=False))
        ])
    
    callbacks += get_default_callbacks(
                    early_stopping_params=early_stopping_params,
                    include_report_wandb=include_report_wandb
                )

    return callbacks


