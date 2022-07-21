import argparse
from calendar import c
import os
import shutil
import re
from datetime import datetime

#####################################################
# Configuration
#####################################################

parser = argparse.ArgumentParser()
parser.add_argument("dataset", metavar="dataset", type=str, help="Dataset parameter search")
parser.add_argument("aggregator", metavar="aggregator", type=str, help="Aggregator type")

args = parser.parse_args()

dataset = args.dataset
aggregator_str = args.aggregator

print(f"Using -- Dataset:{dataset} Aggregator:{aggregator_str}")

#####################################################
# Configuration
#####################################################
CHECKPOINT_BK_DIR = f"{dataset}/{aggregator_str}/checkpoint_"
CHECKPOINT_DIR = f"{dataset}/{aggregator_str}/checkpoint"

if not os.path.exists(CHECKPOINT_BK_DIR):
    print(f"Creating checkpoint backup {CHECKPOINT_BK_DIR} (NOT IN GIT)")
    os.makedirs(CHECKPOINT_BK_DIR)

# Copying not existing files to backup
for path, _, files in os.walk(CHECKPOINT_DIR):
    for file in files:
        in_checkpoint_file = os.path.join(path, file)
        checkpoint_file = in_checkpoint_file.replace(CHECKPOINT_DIR, CHECKPOINT_BK_DIR)
        if not os.path.exists(checkpoint_file):            
            print(f"Copying {in_checkpoint_file} to {checkpoint_file}")
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            shutil.copyfile(in_checkpoint_file, checkpoint_file)

# Cleaning files
sort_regex = re.compile(r"(\d{4}\-\d{2}\-\d{2})_(\d{2}\-\d{2}\-\d{2})")

files_history = {}

for path, _, files in os.walk(CHECKPOINT_DIR):
    for file in files:
        in_checkpoint_file = os.path.join(path, file)

        match = sort_regex.search(in_checkpoint_file)
        if match:
            file_timestamp = datetime.strptime(match.group(0), "%Y-%m-%d_%H-%M-%S")
            file_key = in_checkpoint_file.replace(match.group(0), "~~~")
            files_history[file_key] = files_history.get(file_key, [])
            files_history[file_key].append(file_timestamp)
            files_history[file_key].sort(reverse=True)
            

for file_key in files_history:
    for file_date in files_history[file_key][1:]:
        original_file = file_key.replace("~~~", file_date.strftime("%Y-%m-%d_%H-%M-%S"))
        print(f"Removing {original_file}")
        os.remove(original_file)

