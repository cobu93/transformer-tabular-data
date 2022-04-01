import multiprocessing
import subprocess
import time
import tensorboard
import os
import signal

datasets = ["adult", "jasmine"]
aggregators = ["cls", "concatenate", "rnn", "sum", "mean", "max"]

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
            os.mkdir(f"{dataset}/{aggregator}/checkpoint")

        param_search_process = multiprocessing.Process(
            target=subprocess.check_call,
            args=([f"python param_search.py {dataset} {aggregator}"],),
            kwargs={"shell": True}
        )

        tensorboard_process = multiprocessing.Process(
            target=tensorboard_run,
            args=(f"{dataset}/{aggregator}/checkpoint",)
        )

        param_search_process.start()
        tensorboard_process.start()
        param_search_process.join()
        exit_codes[f"{dataset}.{aggregator}"] = param_search_process.exitcode

        
        if param_search_process.exitcode == 0:
            train_best_process = multiprocessing.Process(
                target=subprocess.check_call,
                args=([f"python train_model.py {dataset} {aggregator}"],),
                kwargs={"shell": True}
            )
        
            train_best_process.start()
            train_best_process.join()
        
        os.kill(tensorboard_process.pid, signal.SIGINT)   


print("The exit codes were:")

for key in exit_codes:
    print(key, "--->", exit_codes[key], f"[{'SUCCESS' if exit_codes[key] == 0 else 'FAILURE' }]")

#subprocess.check_call(
#    ["git add -A && git commit -m 'Automatic uploading after param search execution' && git pull && git push"],
#    shell=True
#)
        