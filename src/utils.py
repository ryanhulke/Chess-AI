import torch
import numpy as np
import torch.nn.functional as F
from dataset import tokenize, MOVE_TO_ACTION, BOARD_STATE_VOCAB_SIZE, get_uniform_buckets_edges_values, extract_rule_states
NUM_RETURN_BUCKETS = 128

@torch.inference_mode()
def compute_expected_value(logits):
    """
    Given logits over NUM_BINS buckets, return the expected value.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # shape: [B, NUM_BINS]
    # Get bucket centers (e.g., uniformly spaced win percentages from 0 to 1)
    _, bucket_values = get_uniform_buckets_edges_values(NUM_RETURN_BUCKETS)
    # Compute expected value (a weighted sum over the bucket centers)
    expected_value = (probs * torch.tensor(bucket_values, device=probs.device, dtype=probs.dtype)).sum(dim=-1)
    return expected_value  # shape: [B]

def get_move(board, model, device, type_casting, include_rule_states=False):
    state_tokens = tokenize(board.fen()).astype(np.int32)  # shape: [state_length]
    legal_moves = list(board.legal_moves)
    sequences = []
    move_candidates = []

    rule_tokens = None
    if include_rule_states:
        repetition_state, move_rule_state = extract_rule_states(board.fen())
        rule_states_offset = BOARD_STATE_VOCAB_SIZE + len(MOVE_TO_ACTION)
        rule_tokens = np.array([
            repetition_state + rule_states_offset,
            move_rule_state + rule_states_offset + 1
        ], dtype=np.int32)

    for move in legal_moves:
        try:
            # each move is represented as a single token. The move token is the index from the mapping,
            # plus an offset equal to BOARD_STATE_VOCAB_SIZE
            action_token = MOVE_TO_ACTION[move.uci()] + BOARD_STATE_VOCAB_SIZE
        except KeyError:
            print(f"Move {move.uci()} not found in MOVE_TO_ACTION mapping; skipping.")
            continue
        
        # construct the input sequence: state tokens + move token + rule tokens (if enabled).
        sequence_parts = [state_tokens, np.array([action_token], dtype=np.int32)]
        if rule_tokens is not None:
            sequence_parts.append(rule_tokens)
        
        seq = np.concatenate(sequence_parts)
        sequences.append(seq)
        move_candidates.append(move)

    if not sequences:
        raise ValueError("No legal moves could be processed; check your MOVE_TO_ACTION mapping.")

    # convert list of sequences into a tensor of shape [B, sequence_length]
    sequences_tensor = torch.tensor(np.stack(sequences), dtype=torch.long).to(device)

    with type_casting, torch.no_grad():
        outputs = model(sequences_tensor)  # shape: [B, sample_sequence_length, NUM_BINS]
    
    # we assume that the move prediction corresponds to the final token in the sequence:
    value_logits = outputs[:, -1, :]  # shape: [B, NUM_BINS]

    # compute expected values for each move
    exp_vals = compute_expected_value(value_logits)  # shape: [B]
    exp_vals = exp_vals.float().cpu().numpy()

    # check if the top 3 moves have the same expected value
    top_moves = np.argsort(exp_vals)[-3:]  # Get indices of the top 3 moves
    if len(set(exp_vals[top_moves])) == 1:
        print("Top 3 moves have the same expected value; selecting randomly among them.")
        # print the number of pieces on the board
        print(f"Number of pieces on the board: {len(board.piece_map())}")
        best_idx = np.random.choice(top_moves)
    else:
        # the model predicts win probability from the current player's perspective,
        # so both White and Black should choose the move with the highest expected value
        best_idx = int(np.argmax(exp_vals))
    
    best_move = move_candidates[best_idx]

    print("Selected move:", best_move.uci(), "with expected value:", exp_vals[best_idx])
    return best_move


def get_best_move_action_value(board, model, device, type_casting, include_rule_states=False):
    """
    Returns the move with the highest expected value.
    """
    return get_move(board, model, device, type_casting, include_rule_states)