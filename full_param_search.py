import multiprocessing
import subprocess
import time
import tensorboard
import os
import signal


# Hecho
datasets = [
    "sylvine", "anneal", "adult", "jasmine", "nomao", "ldpa", "australian", "kr-vs-kp", "volkert"
    ]

aggregators = ["cls", "concatenate", "max", "mean", "rnn", "sum"]


def tensorboard_run(logdir):
    tb = tensorboard.program.TensorBoard()
    tb.configure(bind_all=True, logdir=logdir)
    url = tb.launch()
    print("TensorBoard %s started at %s" % (tensorboard.__version__, url))
    pid = os.getpid()
    print("PID = %d; use 'kill %d' to quit" % (pid, pid))
    while True:
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            break
    print()
    print("Shutting down")

exit_codes = {}

for dataset in datasets:
    for aggregator in aggregators:
        print(f"Trying {dataset}.{aggregator}")

        
        if not os.path.exists(f"{dataset}/{aggregator}/checkpoint"):
            os.makedirs(f"{dataset}/{aggregator}/checkpoint")
        
        # Hyperparameter search
        param_search_process = multiprocessing.Process(
            target=subprocess.check_call,
            args=([f"python param_search.py {dataset} {aggregator}"],),
            kwargs={"shell": True}
        )

        # Tensorboard
        tensorboard_process = multiprocessing.Process(
            target=tensorboard_run,
            args=(f"{dataset}/{aggregator}/checkpoint",)
        )

        param_search_process.start()
        tensorboard_process.start()
        param_search_process.join()
        exit_codes[f"{dataset}.{aggregator}"] = param_search_process.exitcode

        os.kill(tensorboard_process.pid, signal.SIGINT)   

        clean_param_search_process = multiprocessing.Process(
            target=subprocess.check_call,
            args=([f"python clean_files.py {dataset} {aggregator}"],),
            kwargs={"shell": True}
        )

        clean_param_search_process.start()
        clean_param_search_process.join()
        
        evaluate_model_process = multiprocessing.Process(
            target=subprocess.check_call,
            args=([f"python model_evaluation.py {dataset} {aggregator}"],),
            kwargs={"shell": True}
        )

        evaluate_model_process.start()
        evaluate_model_process.join()
        
        evaluate_trials_process = multiprocessing.Process(
            target=subprocess.check_call,
            args=([f"python trials_evaluation.py {dataset} {aggregator}"],),
            kwargs={"shell": True}
        )

        evaluate_trials_process.start()
        evaluate_trials_process.join()


print("The exit codes were:")

for key in exit_codes:
    print(key, "--->", exit_codes[key], f"[{'SUCCESS' if exit_codes[key] == 0 else 'FAILURE' }]")

        