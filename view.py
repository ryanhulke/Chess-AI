import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import chess
import chess.svg
import io
import cairosvg
import torch
import torch.nn.functional as F
import numpy as np
from stockfish import Stockfish
from contextlib import nullcontext
import os
import random

# Import model, configuration, and dataset utilities.
from model import BidirectionalPredictor, PredictorConfig
from dataset import tokenize, get_uniform_buckets_edges_values, BOARD_STATE_VOCAB_SIZE, MOVE_TO_ACTION

# -------------------------------------------------------------------------
# Dummy Implementations & Setup
# -------------------------------------------------------------------------
# For demonstration, we define dummy versions of functions and variables 
# that would normally be imported from your model and dataset modules.
# Replace these with your actual implementations.

NUM_BINS = 128
stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

# Model and evaluation directories.
output_dir = "out"
model_config_path = os.path.join(output_dir, "model_config.json")

eval_model_dir = os.path.join("out", "eval")
model_path = os.path.join(eval_model_dir, "model_hiper.pt")


def get_uniform_buckets_edges_values(num_bins):
    # Returns uniformly spaced edges and bucket centers between 0 and 1.
    edges = np.linspace(0, 1, num_bins + 1)
    values = (edges[:-1] + edges[1:]) / 2.0
    return edges, values


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

# Configure Stockfish for high-Elo analysis.
# Adjust the path to your Stockfish binary as needed.
stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"
high_elo_stockfish = Stockfish(
    path=stockfish_path,
    parameters={"UCI_Elo": 3000}  # High Elo for best move suggestion.
)

# -------------------------------------------------------------------------
# Evaluation Functions
# -------------------------------------------------------------------------

def compute_expected_value(logits):
    """
    Given logits over NUM_BINS buckets, returns the expected value.
    """
    probs = F.softmax(logits, dim=-1)  # shape: [1, NUM_BINS]
    _, bucket_values = get_uniform_buckets_edges_values(NUM_BINS)
    bucket_values_tensor = torch.tensor(bucket_values, device=probs.device, dtype=probs.dtype)
    expected_value = (probs * bucket_values_tensor).sum(dim=-1)
    return expected_value

def get_expected_values_for_moves(board, model, device, type_casting):
    """
    For every legal move in the board, compute its expected value using the model.
    Returns a dictionary mapping move (UCI string) to expected value.
    """
    state_tokens = tokenize(board.fen())
    legal_moves = list(board.legal_moves)
    sequences = []
    move_candidates = []
    for move in legal_moves:
        # Each move is represented by an action token with an offset.
        action_token = MOVE_TO_ACTION[move.uci()] + BOARD_STATE_VOCAB_SIZE
        seq = np.concatenate([state_tokens, np.array([action_token], dtype=np.int32)])
        sequences.append(seq)
        move_candidates.append(move)
    if not sequences:
        return {}
    sequences_tensor = torch.tensor(np.stack(sequences), dtype=torch.long).to(device)
    with type_casting, torch.no_grad():
        outputs = model(sequences_tensor)  # shape: [B, sequence_length, NUM_BINS]
    value_logits = outputs[:, -1, :]  # Evaluate on the final token.
    exp_vals = compute_expected_value(value_logits)  # shape: [B]
    exp_vals = exp_vals.cpu().numpy()
    return {move_candidates[i].uci(): exp_vals[i] for i in range(len(move_candidates))}

def get_best_move_action_value(board, model, device, type_casting):
    """
    Uses the model to determine the best move based on expected values.
    Assumes that higher expected values are better for White.
    Returns the best move and a dictionary of expected values.
    """
    exp_values = get_expected_values_for_moves(board, model, device, type_casting)
    if board.turn:  # White's turn (user).
        best_move_uci = max(exp_values, key=exp_values.get)
    else:  # Black's turn.
        best_move_uci = min(exp_values, key=exp_values.get)
    for move in board.legal_moves:
        if move.uci() == best_move_uci:
            return move, exp_values
    return None, exp_values

def get_stockfish_best_move(board):
    """
    Returns the best move from high-Elo Stockfish.
    """
    high_elo_stockfish.set_fen_position(board.fen())
    best_move = high_elo_stockfish.get_best_move()
    return best_move

def get_board_image(board):
    """
    Renders the board as an image using chess.svg and converts it to a PhotoImage.
    """
    svg_data = chess.svg.board(board=board)
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    return ImageTk.PhotoImage(image)

# -------------------------------------------------------------------------
# Tkinter UI: Chess Analyzer
# -------------------------------------------------------------------------

class ChessAnalyzer(tk.Tk):
    def __init__(self, model, device, type_casting):
        super().__init__()
        self.title("Chess AI Analyzer")
        self.model = model
        self.device = device
        self.type_casting = type_casting
        self.board = chess.Board()  # Start a new game.
        
        # Create two frames: one for the board, one for analysis.
        self.board_frame = tk.Frame(self)
        self.board_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.analysis_frame = tk.Frame(self)
        self.analysis_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)
        
        # Canvas for displaying the board.
        self.canvas = tk.Canvas(self.board_frame, width=480, height=480)
        self.canvas.pack()
        
        # Analysis: Stockfish suggestion.
        self.stockfish_label = tk.Label(self.analysis_frame, text="Stockfish Suggestion: ", font=("Arial", 12))
        self.stockfish_label.pack(pady=5)
        
        # Analysis: Expected values for legal moves.
        self.expected_values_label = tk.Label(self.analysis_frame, text="Expected Values for Legal Moves:", font=("Arial", 12))
        self.expected_values_label.pack(pady=5)
        
        # Listbox to display legal moves with expected values.
        self.move_listbox = tk.Listbox(self.analysis_frame, width=30, height=15)
        self.move_listbox.pack(pady=5)
        self.move_listbox.bind("<Double-Button-1>", self.user_move_selected)
        
        # "Next Move" button to play the next move automatically (for AI moves).
        self.next_move_button = tk.Button(self.analysis_frame, text="Next Move", command=self.next_move)
        self.next_move_button.pack(pady=5)
        
        # Optional manual move entry.
        self.move_entry = tk.Entry(self.analysis_frame)
        self.move_entry.pack(pady=5)
        self.move_entry.bind("<Return>", self.manual_move)
        
        # Info label.
        self.info_label = tk.Label(self.analysis_frame, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)
        
        self.update_board()

    def update_board(self):
        """Refreshes the board image and analysis panel."""
        self.photo = get_board_image(self.board)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.update_analysis()

    def update_analysis(self):
        """Updates Stockfish suggestion and (if it's the user's turn) the expected values for each move."""
        try:
            sf_move = get_stockfish_best_move(self.board)
            self.stockfish_label.config(text=f"Stockfish Suggestion: {sf_move}")
        except Exception as e:
            self.stockfish_label.config(text=f"Stockfish error: {e}")
        
        if self.board.turn:
            # It's the user's turn (playing White).
            exp_values = get_expected_values_for_moves(self.board, self.model, self.device, self.type_casting)
            self.move_listbox.delete(0, tk.END)
            # Sort moves by expected value (highest first).
            sorted_moves = sorted(exp_values.items(), key=lambda x: x[1], reverse=True)
            for move_uci, val in sorted_moves:
                self.move_listbox.insert(tk.END, f"{move_uci}: {val:.3f}")
            self.info_label.config(text="Your turn: double-click a move or type it in the box.")
        else:
            # AI's turn.
            self.move_listbox.delete(0, tk.END)
            self.info_label.config(text="AI's turn: press 'Next Move' to continue.")

    def next_move(self):
        """
        Plays the next move automatically when it's the AI's turn.
        If it is the user's turn, shows an info message.
        """
        if not self.board.turn:
            move, _ = get_best_move_action_value(self.board, self.model, self.device, self.type_casting)
            if move is not None and move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
            else:
                messagebox.showerror("Error", "AI could not determine a move.")
        else:
            messagebox.showinfo("Info", "It's your turn. Please select a move from the list or enter a move manually.")

    def user_move_selected(self, event):
        """
        Called when the user double-clicks a move from the listbox.
        The selected move is played.
        """
        selection = self.move_listbox.curselection()
        if selection:
            move_text = self.move_listbox.get(selection[0]).split(":")[0]
            try:
                move = chess.Move.from_uci(move_text)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board()
                else:
                    messagebox.showerror("Illegal Move", f"{move_text} is not legal.")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid move: {e}")

    def manual_move(self, event):
        """
        Called when the user types a move in the entry box and presses Enter.
        """
        move_text = self.move_entry.get().strip()
        try:
            move = chess.Move.from_uci(move_text)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_entry.delete(0, tk.END)
                self.update_board()
            else:
                messagebox.showerror("Illegal Move", f"{move_text} is not legal.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid move: {e}")

# -------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------
if __name__ == '__main__':
    app = ChessAnalyzer(model, device, type_casting)
    app.mainloop()
