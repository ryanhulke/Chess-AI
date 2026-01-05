import torch
import torch.nn.functional as F
import numpy as np
import os
import chess
from contextlib import nullcontext
from stockfish import Stockfish
from model import BidirectionalPredictor, PredictorConfig
from dataset import tokenize, get_uniform_buckets_edges_values, BOARD_STATE_VOCAB_SIZE, MOVE_TO_ACTION

NUM_BINS = 128

stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

# Model and evaluation directories.
output_dir = "model"
model_config_path = os.path.join(output_dir, "model_config.json")

eval_model_dir = "model"
model_path = os.path.join(eval_model_dir, "model_gated.pt")

# Load your evaluation model.
if os.path.exists(model_path):
    eval_model_config = PredictorConfig.from_json(model_config_path)
    # Load main model.
    model = BidirectionalPredictor(eval_model_config)
    model.load_state_dict(torch.load(model_path))
else:
    raise ValueError("Model file does not exist")

if torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'
device = torch.device(device_type)
print(f"Device: {device_type}")

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
type_casting = nullcontext() if device_type in {'cpu', 'mps'} else torch.amp.autocast('cuda', dtype=ptdtype)
print(f"Type: {dtype}")

model.to(device)
model.eval()  # Set to evaluation mode.

def compute_expected_value(logits):
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # shape: [1, NUM_BINS]
    _, bucket_values = get_uniform_buckets_edges_values(NUM_BINS)
    expected_value = (probs * torch.tensor(bucket_values, device=probs.device, dtype=probs.dtype)).sum(dim=-1)
    return expected_value  # shape: [1]

def get_best_move_action_value(board, model, device, type_casting, additional_registers=2):
    state_tokens = tokenize(board.fen()).astype(np.int32)  # shape: [state_length]
    legal_moves = list(board.legal_moves)
    sequences = []
    move_candidates = []

    for move in legal_moves:
        try:
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
    # move prediction is the final token in the sequence
    value_logits = outputs[:, -1, :]  # shape: [B, NUM_BINS]
    exp_vals = compute_expected_value(value_logits)  # shape: [B]
    exp_vals = exp_vals.float().cpu().numpy()

    # For white, select the move with the highest expected win percentage;
    # for black, select the move with the lowest expected win percentage.
    best_idx = int(np.argmax(exp_vals)) if board.turn else int(np.argmin(exp_vals))
    best_move = move_candidates[best_idx]

    # print("Selected move:", best_move.uci(), "with expected value:", exp_vals[best_idx])
    return best_move

def play_game(model, device, opponent_rating=1500, ai_rating=1500, k_factor=10):
    opponent_model = Stockfish(
    path=stockfish_path,
    parameters={"UCI_Elo": opponent_rating}
    )
    board = chess.Board()
    while not board.is_game_over():
        if board.turn:  # White
            move = get_best_move_action_value(board, model, device, type_casting)
            # print("user move:", move)
        else:
            move = opponent_model.get_best_move()
            move = chess.Move.from_uci(move)
            # print("stockfish move:", move)

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
    
    # determine score S for your AI (playing white)
    if result == "1-0":
        S = 1.0
    elif result == "0-1":
        S = 0.0
    elif result == "1/2-1/2":
        S = 0.5
    else:
        S = 0.5  # default to draw for unexpected result
    
    # Calculate expected score E using the Elo formula
    E = 1 / (1 + 10 ** ((opponent_rating - ai_rating) / 400))
    new_ai_rating = ai_rating + k_factor * (S - E)
    return new_ai_rating, result


def main():
    opponent_rating = 300
    ai_rating = 2100  # initial rating for AI
    num_games = 6  # Number of evaluation game
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