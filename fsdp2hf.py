from typing import List, Tuple, Dict
import re
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
# Make sure torch.distributed is available if DTensor is used directly
try:
    from torch.distributed._tensor import DTensor, Shard, Placement
    TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    print("Warning: torch.distributed._tensor not found. DTensor merging might fail.")
    TORCH_DISTRIBUTED_AVAILABLE = False
    # Define dummy classes if needed, or handle error later
    class DTensor: pass
    class Placement: pass
    class Shard(Placement): pass


from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    """Merges tensors based on their DTensor placement."""
    if not TORCH_DISTRIBUTED_AVAILABLE:
         raise RuntimeError("torch.distributed is required for DTensor merging.")

    # Use isinstance checks as placement might be Shard, Replicate, Partial etc.
    # Accessing placement attributes like is_replicate() might be cleaner if available
    # Assuming Placement is the base class or has ways to identify type

    if isinstance(placement, Shard): # Check if it's specifically a Shard placement
        if hasattr(placement, 'dim'): # Ensure it has the 'dim' attribute
             print(f"Merging Sharded tensor along dim: {placement.dim}")
             # Ensure tensors are on the same device before cat, ideally CPU
             tensors_on_cpu = [t.cpu() for t in tensors]
             return torch.cat(tensors_on_cpu, dim=placement.dim).contiguous()
        else:
             raise ValueError("Shard placement object does not have 'dim' attribute.")
    # Simple check for replicate - assumes first tensor is representative
    elif len(tensors) > 0 and tensors[0].size() == tensors[-1].size(): # Heuristic for replicate/partial
         print("Assuming Replicate placement, returning first tensor.")
         return tensors[0].cpu().contiguous() # Return a copy on CPU
    # Add checks for Replicate() and Partial() if specific classes exist and are used
    # elif isinstance(placement, Replicate):
    #     return tensors[0]
    # elif isinstance(placement, Partial):
    #     raise NotImplementedError("Partial placement merging is not supported yet")
    else:
        raise ValueError(f"Unsupported or ambiguous placement type for merging: {type(placement)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge distributed PyTorch checkpoints into a Hugging Face model.")
    parser.add_argument('--local_dir', type=str, required=True,
                        help='Absolute path to the directory containing the sharded model checkpoints (e.g., model_world_size_X_rank_Y.pt).')
    parser.add_argument('--model_src_dir', type=str, required=False, default="/home/wangyc/verl/Qwen/Qwen2.5-7B-Instruct",
                        help='Absolute path to the directory containing the original Hugging Face model config and tokenizer files.')
    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='Absolute path to the directory where the merged model will be saved. If not provided, the parent directory of local_dir will be used.')

    args = parser.parse_args()

    local_dir = args.local_dir
    model_src_dir = args.model_src_dir
    save_dir = args.save_dir
    # --- Validation ---

    if not os.path.isdir(local_dir):
        raise NotADirectoryError(f"Provided local_dir does not exist or is not a directory: {local_dir}")
    if not os.path.isdir(model_src_dir):
         raise NotADirectoryError(f"Provided model_src_dir does not exist or is not a directory: {model_src_dir}")

    # --- Generate Output Path ---
    if save_dir == None:
        parent_dir = os.path.dirname(local_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        output_path = os.path.join(grandparent_dir, "hf")
    else:
        output_path = args.save_dir
    # model_identifier = os.path.basename(local_dir) # Use the last part of local_dir for distinction
    # output_path = os.path.join(base_output_dir, model_identifier)
    print(f"Input directory (shards): {local_dir}")
    print(f"Source directory (config/tokenizer): {model_src_dir}")
    print(f"Output directory (merged model): {output_path}")
    os.makedirs(output_path, exist_ok=True) # Ensure the final output directory exists


    # --- Prepare Hugging Face config/tokenizer in a temporary location within local_dir ---
    hf_path = os.path.join(local_dir, 'huggingface_temp_config') # Use a distinct temp name
    os.makedirs(hf_path, exist_ok=True)

    print(f"Copying config and tokenizer files from {model_src_dir} to {hf_path}")

    # Copy config.json
    config_src = os.path.join(model_src_dir, "config.json")
    config_dst = os.path.join(hf_path, "config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, config_dst)
        print(f"Copied config.json")
    else:
        raise FileNotFoundError(f"config.json not found in {model_src_dir}")

    # Copy tokenizer related files
    # Include common tokenizer files, adapt if your model uses different ones
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json", # For some tokenizers
        "merges.txt", # For some tokenizers
        "vocab.txt",  # For BERT-like tokenizers
        "sentencepiece.bpe.model", # For SentencePiece
        "added_tokens.json" # If present
    ]

    copied_tokenizer_files = False
    for file in tokenizer_files:
        src_file = os.path.join(model_src_dir, file)
        dst_file = os.path.join(hf_path, file)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {file}")
            copied_tokenizer_files = True

    if not copied_tokenizer_files:
         print(f"Warning: No tokenizer files found/copied from {model_src_dir}. Ensure the necessary files exist.")
         # Consider making this an error if tokenizer is strictly required downstream

    # --- Load and Merge Shards ---
    print("Scanning for model shards...")
    world_size = 0
    rank0_file = None
    # Find rank 0 file to determine world size and mesh info
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1)) # Convert to int
            rank0_file = filename
            print(f"Found rank 0 file: {rank0_file}, world size: {world_size}")
            break
    if not rank0_file:
        raise FileNotFoundError(f"Could not find rank 0 model file (model_world_size_*_rank_0.pt) in {local_dir}")

    # Load rank 0 state dict to get mesh info
    print("Loading rank 0 state dict to determine sharding...")
    rank0_path = os.path.join(local_dir, rank0_file)
    state_dict_rank0 = torch.load(rank0_path, map_location='cpu', weights_only=False) # Use weights_only=False for DTensors

    # Find a DTensor key to inspect placements
    dtensor_key = None
    dtensor_instance = None
    if TORCH_DISTRIBUTED_AVAILABLE:
        for key, value in state_dict_rank0.items():
            if isinstance(value, DTensor):
                dtensor_key = key
                dtensor_instance = value
                break

    if not dtensor_key:
         # If no DTensor found, maybe it's a different format or already merged?
         # For now, assume standard tensors if no DTensor is present.
         print("Warning: No DTensor instances found in rank 0 state_dict. Assuming standard tensors.")
         # Simplified logic for non-DTensor scenario
         merged_state_dict = state_dict_rank0 # Directly use rank 0 if no DTensors
         total_shards = 1 # Treat as single shard
    else:
         print(f"Inspecting DTensor key: {dtensor_key}")
         device_mesh = dtensor_instance.device_mesh
         mesh = device_mesh.mesh
         mesh_dim_names = device_mesh.mesh_dim_names # Often None or ('data', 'model') etc.

         print(f'Device Mesh: {mesh}, Mesh Dim Names: {mesh_dim_names}') # Mesh dim names might be None

         # Determine total shards based on world size found in filename
         total_shards = world_size
         # mesh_shape might not be directly deducible without more context on how sharding was done
         # Rely on world_size from filename as the primary indicator of shard count
         print(f'Processing {total_shards} model shards based on detected world size.')

         # --- Load all shards ---
         print(f"Loading {total_shards} shards...")
         model_state_dict_lst = [None] * total_shards # Initialize with None
         model_state_dict_lst[0] = state_dict_rank0 # Already loaded

         def process_one_shard(rank):
             model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
             if not os.path.exists(model_path):
                 print(f"Warning: Shard file not found for rank {rank}: {model_path}")
                 return None # Return None if file missing
             print(f"Loading shard rank {rank}...")
             try:
                 state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                 model_state_dict_lst[rank] = state_dict
                 print(f"Loaded shard rank {rank}")
             except Exception as e:
                 print(f"Error loading shard rank {rank} from {model_path}: {e}")
                 # Keep model_state_dict_lst[rank] as None

         # Use ThreadPoolExecutor for parallel loading
         with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor: # Limit workers
             futures = [executor.submit(process_one_shard, rank) for rank in range(1, total_shards)]
             # Wait for all loading tasks to complete
             for future in futures:
                  future.result() # Raise exceptions if any occurred during loading

         # Filter out None entries if some shards failed to load
         loaded_state_dicts = [sd for sd in model_state_dict_lst if sd is not None]
         if len(loaded_state_dicts) != total_shards:
              print(f"Warning: Expected {total_shards} shards, but only loaded {len(loaded_state_dicts)}. Proceeding with loaded shards.")
              # Adjust total_shards if necessary, or handle error depending on requirements
              # total_shards = len(loaded_state_dicts) # Optional: Adjust if proceeding

         if not loaded_state_dicts:
              raise RuntimeError("Failed to load any model state dicts.")

         # --- Consolidate tensors from all loaded shards ---
         print("Consolidating tensors from shards...")
         consolidated_tensors: Dict[str, List[torch.Tensor]] = {}
         param_placements: Dict[str, Tuple[Placement]] = {} # Store placements per parameter key
         # Use keys from rank 0 as the reference
         reference_keys = set(state_dict_rank0.keys())

         for key in reference_keys:
              consolidated_tensors[key] = []
              first_tensor_type = None
              placement_info = None

              for i, model_state_dict in enumerate(loaded_state_dicts):
                  if key not in model_state_dict:
                      print(f"Warning: Key '{key}' not found in shard {i}. Skipping.")
                      continue # Skip this shard for this key

                  tensor = model_state_dict[key] # No pop, just access

                  # Store placement info from the first encountered DTensor for this key
                  if isinstance(tensor, DTensor) and key not in param_placements:
                       placements = tuple(tensor.placements)
                       param_placements[key] = placements
                       placement_info = placements # Store for validation
                       first_tensor_type = DTensor

                  # Validate placements for subsequent shards
                  if isinstance(tensor, DTensor) and key in param_placements:
                       current_placements = tuple(tensor.placements)
                       if current_placements != placement_info:
                            print(f"Warning: Inconsistent placements for key '{key}' across shards. Expected {placement_info}, got {current_placements} in shard {i}. Using first placement.")
                            # Decide how to handle inconsistency - here we trust the first shard's placement

                  # Extract local tensor for DTensor or use tensor directly
                  if isinstance(tensor, DTensor):
                       local_tensor = tensor.to_local() # Use to_local() for DTensor
                       consolidated_tensors[key].append(local_tensor.to(torch.bfloat16).cpu()) # Move to CPU
                  else:
                       # Handle non-DTensor tensors (e.g., scalars, buffers)
                       if first_tensor_type is None:
                           first_tensor_type = type(tensor)
                       elif first_tensor_type != type(tensor):
                           print(f"Warning: Inconsistent tensor types for key '{key}' across shards.")

                       # Add non-DTensor directly, assuming it should be replicated/identical
                       # Only add once if it's not expected to be sharded
                       if i == 0: # Add non-DTensor only from the first shard
                           consolidated_tensors[key] = tensor.to(torch.bfloat16).cpu() # Convert and move to CPU
                           param_placements[key] = None # Indicate no sharding placement needed
                       break # Stop processing other shards for this non-DTensor key

         # Clear memory
         del model_state_dict_lst
         del loaded_state_dicts
         del state_dict_rank0
         import gc
         gc.collect()

         # --- Merge the consolidated tensors ---
         print("Merging consolidated tensors...")
         merged_state_dict = {}
         for key in sorted(consolidated_tensors.keys()):
             tensor_data = consolidated_tensors[key]
             placements = param_placements.get(key) # Get placement for this key

             if isinstance(tensor_data, list): # Needs merging
                 if not tensor_data:
                      print(f"Warning: No tensors collected for key '{key}'. Skipping.")
                      continue

                 if placements is None or not TORCH_DISTRIBUTED_AVAILABLE:
                      # If placements not determined (non-DTensor) or torch.distributed unavailable,
                      # assume replicated/identical - take the first tensor.
                      print(f"Key '{key}': Assuming replicated/no sharding. Taking tensor from first shard.")
                      merged_state_dict[key] = tensor_data[0] # Already on CPU and bf16
                 else:
                     # We have a list of tensors and placement info for DTensors
                     print(f"Merging key '{key}' with {len(tensor_data)} pieces. Placements: {placements}")
                     # Assume simple FSDP for now if mesh analysis is complex
                     # Use the first placement dimension for merging logic
                     if len(placements) > 0:
                          # Pass only the relevant placement dimension for merging
                          try:
                              merged_state_dict[key] = merge_by_placement(tensor_data, placements[0]) # Pass first placement
                          except Exception as e:
                              print(f"Error merging key '{key}' with placement {placements[0]}: {e}")
                              # Fallback or skip key
                              print(f"Skipping key '{key}' due to merging error.")
                              continue
                     else:
                          print(f"Warning: No placement info found for sharded key '{key}'. Taking first piece.")
                          merged_state_dict[key] = tensor_data[0] # Fallback: take first piece

             else: # Already a single tensor (non-DTensor case)
                 print(f"Key '{key}': Already consolidated. Using directly.")
                 merged_state_dict[key] = tensor_data # Already on CPU and bf16

    # --- Load model structure and save ---
    print('Loading model structure from config...')
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)

    # Determine model type from config
    auto_model_class = AutoModelForCausalLM # Default assumption
    if config.architectures:
        if 'ForTokenClassification' in config.architectures[0]:
            auto_model_class = AutoModelForTokenClassification
        elif 'ForCausalLM' in config.architectures[0]:
            auto_model_class = AutoModelForCausalLM
        # Add other model types as needed
        else:
             print(f'Warning: Unknown architecture {config.architectures[0]}. Defaulting to AutoModelForCausalLM.')
             # raise NotImplementedError(f'Unknown architecture {config.architectures}') # Or raise error
    else:
        print("Warning: config.architectures not found. Defaulting to AutoModelForCausalLM.")


    # Load model on meta device then assign weights
    try:
        print(f"Loading model structure using {auto_model_class.__name__}...")
        with torch.device('meta'):
            model = auto_model_class.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)

        # Materialize model on CPU and load state dict
        # model.to_empty(device='cpu') # This might not be needed if loading directly
        print("Loading merged state dict into model structure...")
        # Strict=False allows loading even if some keys mismatch (e.g., buffer vs parameter)
        missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False, assign=True)
        if missing_keys:
            print("Warning: Missing keys during state dict load:")
            print(missing_keys)
        if unexpected_keys:
            print("Warning: Unexpected keys found in state dict:")
            print(unexpected_keys)

    except Exception as e:
         print(f"Error loading model structure or state dict: {e}")
         print("Attempting to save state_dict directly without model structure loading.")
         # Fallback: Save just the state dict if model loading fails
         state_dict_save_path = os.path.join(output_path, "pytorch_model.bin") # Standard HF name
         torch.save(merged_state_dict, state_dict_save_path)
         print(f"Saved merged state_dict directly to {state_dict_save_path}")
         # Try saving config and tokenizer anyway
         try:
             config.save_pretrained(output_path)
             tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
             tokenizer.save_pretrained(output_path)
         except Exception as te:
             print(f"Error saving config/tokenizer after fallback: {te}")
         exit() # Exit after fallback save


    # --- Save Model and Tokenizer ---
    print(f'Saving merged model to {output_path}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path) # save_pretrained handles state dict saving internally
        print(f"Model and tokenizer successfully saved to {output_path}")
    except Exception as e:
         print(f"Error saving model/tokenizer to {output_path}: {e}")
         print("Attempting to save state_dict directly as fallback.")
         state_dict_save_path = os.path.join(output_path, "pytorch_model.bin")
         torch.save(merged_state_dict, state_dict_save_path)
         print(f"Saved merged state_dict directly to {state_dict_save_path}")


    # --- Cleanup ---
    print(f"Cleaning up temporary config directory: {hf_path}")
    try:
        shutil.rmtree(hf_path)
    except OSError as e:
        print(f"Error removing temporary directory {hf_path}: {e}")

    print("Script finished.")