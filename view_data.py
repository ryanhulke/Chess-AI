import os
from os import listdir
from os.path import isfile, join
from dataset import ActionValueDataset
import torch
import torch.nn.functional as F
import torch.cuda.amp
import numpy as np
import os
import random
import chess
from contextlib import nullcontext
from stockfish import Stockfish

# Import model, configuration, and dataset utilities.
from model import BidirectionalPredictor, PredictorConfig
from dataset import tokenize, get_uniform_buckets_edges_values, BOARD_STATE_VOCAB_SIZE, MOVE_TO_ACTION
# Also define NUM_BINS as your number of value bins (e.g., 128)
NUM_BINS = 128

stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

# Model and evaluation directories.
output_dir = "out"
model_config_path = os.path.join(output_dir, "model_config.json")

eval_model_dir = os.path.join("out", "train")
model_path = os.path.join(eval_model_dir, "model.pt")

# Load your evaluation model.
if os.path.exists(model_path):
    eval_model_config = PredictorConfig.from_json(model_config_path)
    # Load main model.
    model = BidirectionalPredictor(eval_model_config)
    model.load_state_dict(torch.load(model_path))
else:
    raise ValueError("Model file does not exist")

# Setup device.
if torch.backends.mps.is_available():
    device_type = 'mps'
elif torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'
device = torch.device(device_type)
print(f"Device: {device_type}")

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
type_casting = nullcontext() if device_type in {'cpu', 'mps'} else torch.amp.autocast('cuda', dtype=ptdtype)
print(f"Type: {dtype}")
print(f"Using autocast: {device_type not in {'cpu', 'mps'}}")

# Setup gradient scaler for mixed precision if needed.
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

model.to(device)
model.eval()  # Set to evaluation mode.

def main():
    # Directory where training data is stored
    train_data_dir = os.path.join("data", "train")
    # Collect files that start with "action_value"
    train_files = [os.path.join(train_data_dir, f) for f in listdir(train_data_dir) 
                   if isfile(join(train_data_dir, f)) and f.startswith("action_value")]
    
    # Use a small fraction to avoid loading the whole dataset (e.g., 0.001 or 0.01)
    fraction = 0.00001
    
    # Create the dataset with hl_gauss=True and no additional registers (set registers=0 for simplicity)
    dataset = ActionValueDataset(train_files, hl_gauss=True, registers=0, fraction=fraction)
    
    print("Dataset length (after fraction):", len(dataset))
    # Print the first 10 samples (state sequence and target label)
    for i in range(min(3, len(dataset))):
        sample, target = dataset[i]
        print(f"\nSample {i}:")
        print("State+Action tokens:", sample)
        print("Target:", target)
        sample_tensor = torch.tensor(np.stack(sample), dtype=torch.long).unsqueeze(0).to(device)
        with type_casting, torch.no_grad():

            outputs = model(sample_tensor)  # shape: [B, sample_sequence_length, NUM_BINS]
        
        # We assume that the move prediction is the last token in the sequence:
        value_logits = F.softmax(outputs[:, -1, :])  # shape: [B, NUM_BINS]
        print("log:", value_logits)
        
if __name__ == '__main__':
    main()