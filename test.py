from stockfish import Stockfish

# Initialize Stockfish with a specified depth and target Elo rating
stockfish_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

stockfish_engine = Stockfish(
    path=stockfish_path,
    parameters={"UCI_Elo": 2000}  # simulate a 1400-rated opponent
)

print(stockfish_engine.get_best_move())