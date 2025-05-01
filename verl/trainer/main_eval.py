# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local


def get_custom_reward_fn(config):
    import importlib.util
    import os
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = []
    acc_reward_lst = []
    for r in response_lst:
        result = reward_fn(data_source, r, ground_truth)
        # Ensure result is a dict and contains the necessary keys
        if isinstance(result, dict):
            # Append the main score (assuming it's the top-level 'score' key or the dict itself if score is the primary return)
            # Adjust this line based on how your reward_fn returns the main score
            main_score = result.get("score", 0.0) # Example: get 'score' key, default to 0.0 if not found
            score_lst.append(main_score)

            # Extract acc_reward from extra_info
            if "extra_info" in result and "acc_reward" in result["extra_info"]:
                acc_reward_lst.append(result["extra_info"]["acc_reward"])
            else:
                acc_reward_lst.append(0.0) # Default if acc_reward is missing
        else:
             # Handle cases where result is not a dict (e.g., just a float score)
             # In this case, we might only have the main score
             score_lst.append(float(result)) # Assume result is the main score
             acc_reward_lst.append(0.0) # No answer_score available

    # Return data_source, mean of main scores, mean of answer rewards
    return data_source, np.mean(score_lst), np.mean(acc_reward_lst)

@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    # Use two dictionaries to store the scores separately
    data_source_total_reward = defaultdict(list)
    data_source_acc_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [
        process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    print("Processing results...") # Added print statement
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                try:
                    # Unpack both scores
                    data_source, score, answer_score = ray.get(result_id)
                    # Append scores to their respective dictionaries
                    data_source_total_reward[data_source].append(score)
                    data_source_acc_reward[data_source].append(answer_score)
                except Exception as e:
                    print(f"Error processing result {result_id}: {e}") # Add error handling for ray.get
                pbar.update(1)

    # Calculate and store metrics for both scores
    metric_dict = {}
    print("\nCalculating final metrics...") # Added print statement

    # Calculate mean total score
    for data_source, scores in data_source_total_reward.items():
        if scores: # Ensure list is not empty
             metric_dict[f"test_score/{data_source}"] = np.mean(scores)
        else:
             metric_dict[f"test_score/{data_source}"] = 0.0 # Or handle as appropriate

    # Calculate mean answer score
    for data_source, answer_scores in data_source_acc_reward.items():
         if answer_scores: # Ensure list is not empty
             metric_dict[f"test_answer_score/{data_source}"] = np.mean(answer_scores)
         else:
              metric_dict[f"test_answer_score/{data_source}"] = 0.0 # Or handle as appropriate


    print("\n--- Evaluation Metrics ---") # Updated print message
    # Print metrics in a more readable format
    for key, value in metric_dict.items():
        print(f"{key}: {value:.4f}") # Format to 4 decimal places


if __name__ == "__main__":
    main()
