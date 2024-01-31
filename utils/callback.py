
import skorch
import wandb
    
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
        
        wandb.log(self.metrics)

def get_default_callbacks(multiclass=False, include_report_wandb=True):

    callbacks = [
        ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
        ("accuracy", skorch.callbacks.EpochScoring("accuracy", lower_is_better=False)),
        ("duration", skorch.callbacks.EpochTimer())
        ]

    if not multiclass:
        callbacks.extend([
            ("roc_auc", skorch.callbacks.EpochScoring("roc_auc", lower_is_better=False)),
            ("f1", skorch.callbacks.EpochScoring("f1", lower_is_better=False)),
            ("precision", skorch.callbacks.EpochScoring("precision", lower_is_better=False)),
            ("recall", skorch.callbacks.EpochScoring("recall", lower_is_better=False))
        ])

    if include_report_wandb:
        callbacks.append(
            ("report_tune", ReportWandB())
        )

    return callbacks

