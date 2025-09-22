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

eval_model_dir = os.path.join("out", "eval")
model_path = os.path.join(eval_model_dir, "model3.pt")

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

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def compute_expected_value(logits):
    """
    Given logits over NUM_BINS buckets, return the expected value.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # shape: [1, NUM_BINS]
    # Get bucket centers (e.g., uniformly spaced win percentages from 0 to 1)
    _, bucket_values = get_uniform_buckets_edges_values(NUM_BINS)
    # Compute expected value (a weighted sum over the bucket centers)
    expected_value = (probs * torch.tensor(bucket_values, device=probs.device, dtype=probs.dtype)).sum(dim=-1)
    return expected_value  # shape: [1]

def get_best_move_action_value(board, model, device, type_casting, additional_registers=2):
    """
    For a given chess board, evaluate Q(s, a) for every legal move using the action-value predictor.
    The input sequence is constructed by concatenating the tokenized board state, a move token,
    and additional register tokens (as used during training).
    Returns the move with the highest expected value (if white) or lowest (if black).
    """
    # Tokenize the board state (FEN) into a fixed-length sequence.
    state_tokens = tokenize(board.fen()).astype(np.int32)  # shape: [state_length]
    legal_moves = list(board.legal_moves)
    sequences = []
    move_candidates = []

    # For register tokens, compute the register offset as in training.
    # action_offset is BOARD_STATE_VOCAB_SIZE.
    # NUM_ACTIONS is defined as len(MOVE_TO_ACTION)
    NUM_ACTIONS = len(MOVE_TO_ACTION)

    for move in legal_moves:
        try:
            # Each move is represented as a single token. The move token is the index from the mapping,
            # plus an offset equal to BOARD_STATE_VOCAB_SIZE.
            action_token = MOVE_TO_ACTION[move.uci()] + BOARD_STATE_VOCAB_SIZE
        except KeyError:
            print(f"Move {move.uci()} not found in MOVE_TO_ACTION mapping; skipping.")
            continue
        # Construct the input sequence: state tokens + move token.
        seq = np.concatenate([
            state_tokens,
            np.array([action_token], dtype=np.int32),
        ])
        sequences.append(seq)
        move_candidates.append(move)

    if not sequences:
        raise ValueError("No legal moves could be processed; check your MOVE_TO_ACTION mapping.")

    # Convert list of sequences into a tensor of shape [B, sequence_length]
    sequences_tensor = torch.tensor(np.stack(sequences), dtype=torch.long).to(device)

    with type_casting, torch.no_grad():
        outputs = model(sequences_tensor)  # shape: [B, sample_sequence_length, NUM_BINS]
    # We assume that the move prediction corresponds to the final token in the sequence:
    value_logits = outputs[:, -1, :]  # shape: [B, NUM_BINS]
    exp_vals = compute_expected_value(value_logits)  # shape: [B]
    exp_vals = exp_vals.float().cpu().numpy()

    # For white, select the move with the highest expected win percentage;
    # for black, select the move with the lowest expected win percentage.
    best_idx = int(np.argmax(exp_vals)) if board.turn else int(np.argmin(exp_vals))
    best_move = move_candidates[best_idx]

    # print("Selected move:", best_move.uci(), "with expected value:", exp_vals[best_idx])
    return best_move




# -----------------------------------------------------------------------------
# Game Playing and Elo Calculation
# -----------------------------------------------------------------------------

def play_game(model, device, opponent_rating=1500, ai_rating=1500, k_factor=10):
    """
    Simulate a single game between your AI (as white) and an opponent.
    The opponent can be either a model or a random move selector.
    
    Args:
        model (BidirectionalPredictor): Your evaluation model.
        device (torch.device): Device used for inference.
        opponent_model: (Optional) A model for the opponent.
        opponent_rating (int): Elo rating of the opponent.
        ai_rating (int): Current Elo rating of your AI.
        k_factor (int): Elo update constant.
    
    Returns:
        new_ai_rating (float): Updated Elo rating for your AI.
        result (str): Game result ("1-0", "0-1", or "1/2-1/2").
    """
    opponent_model = Stockfish(
    path=stockfish_path,
    parameters={"UCI_Elo": opponent_rating}  # simulate a 1400-rated opponent
    )
    board = chess.Board()
    while not board.is_game_over():
        if board.turn:  # White (your AI) to move.
            move = get_best_move_action_value(board, model, device, type_casting)
            # print("user move:", move)
        else:
            move = opponent_model.get_best_move()
            move = chess.Move.from_uci(move)
            # print("stockfish move:", move)
        # Validate and push move.
        if move in board.legal_moves:
            board.push(move)
            opponent_model.set_fen_position(board.fen())
        else:
            print(board.legal_moves)
            print(move)
            # Illegal move: count as a loss.
            print("Illegal move encountered. Forfeiting game.")
            board = chess.Board()  # reset board state
            return ai_rating - k_factor, "ILLEGAL"
    
    result = board.result()  # e.g. "1-0", "0-1", or "1/2-1/2"
    
    # Determine score S for your AI (playing white).
    if result == "1-0":
        S = 1.0
    elif result == "0-1":
        S = 0.0
    elif result == "1/2-1/2":
        S = 0.5
    else:
        S = 0.5  # default to draw for unexpected result
    
    # Calculate expected score E using the Elo formula.
    E = 1 / (1 + 10 ** ((opponent_rating - ai_rating) / 400))
    new_ai_rating = ai_rating + k_factor * (S - E)
    return new_ai_rating, result

# -----------------------------------------------------------------------------
# Main Evaluation Loop
# -----------------------------------------------------------------------------

def main():
    # Example: Optionally load an opponent model from a different folder.
    # Suppose you maintain separate folders for different rated models.
    # For instance, an opponent model rated 1700.
    
    opponent_rating = 2100


    ai_rating = 2100  # initial rating for your AI.
    num_games = 6  # Number of evaluation games.
    results = []
    
    for i in range(5):
        for j in range(num_games):
            ai_rating, result = play_game(model, device, opponent_rating=opponent_rating, ai_rating=ai_rating)
            results.append(result)
            print(f"Game {(j+1)}: {result}, New AI rating: {ai_rating:.2f}")
        opponent_rating += 50
    print(f"\nFinal estimated Elo for your AI: {ai_rating:.2f}")

if __name__ == '__main__':
    main()
