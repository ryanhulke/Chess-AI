# host the model on a local server with FastAPI

import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
from contextlib import nullcontext
from utils import get_best_move_action_value
from model import PredictorConfig, BidirectionalPredictor

app = FastAPI(title="Transformer-Chess API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device_type = 'mps'
elif torch.cuda.is_available():
    device_type = 'cuda'
else:
    device_type = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
type_casting = nullcontext() if device_type in {'cpu', 'mps'} else torch.amp.autocast(device_type, dtype=ptdtype)
print(f"Device: {device_type}")
print(f"Type: {dtype}")

# load model configuration and weights
config_path = "model/model_config.json"
model_path = "model/final_model.pt"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Model config not found at {config_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

# load configuration and create model
config = PredictorConfig.from_json(config_path)
model = BidirectionalPredictor(config)

# load the trained weights
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print(f"Model loaded successfully with {model.get_num_params()/1e6:.2f}M parameters")


class FENReq(BaseModel):
    fen: str

class MoveResp(BaseModel):
    uci: str

@app.post("/get-move", response_model=MoveResp)
def route_best_move(req: FENReq):
    try:
        board = chess.Board(req.fen)
        move = get_best_move_action_value(board, model, device, type_casting).uci()
        print(f"Predicted best move: {move} for position: {req.fen}")
        return {"uci": move}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Transformer-Chess API is running"}
