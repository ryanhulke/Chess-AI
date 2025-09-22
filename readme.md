## Architecture Overview
- **Bidirectional Transformer encoder** with rotary positional embeddings trained to predict an action-value for a potential move on a given chess board.
- FEN tokenizer: fixed 77-token state - side-to-move (1) + 64 squares + castling rights (4) + en-passant square/file (2) + halfmove clock (3) + fullmove number (3).
- Training sequence: [state tokens, action token, value-bucket token]. Loss is only applied to the final token.
- Targets: win_prob discretized into 128 uniform bins over [0,1], smoothed with a half-life Gaussian.
- Vocab dictionary: board-state alphabet + full precomputed move set + value-bucket classes.
- Inference: For each legal move available, run inference with the model and select the highest valued move.

## Setup:

Clone the repository:

```bash
git clone https://github.com/ryanhulke/Chess-AI
cd Chess-AI
```

### Requirements:

Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

### Download Data:

To download the data, run the following command:

```bash
cd data
./download.sh
```

## Training:

To train the model, run the following command:

```bash
python train.py
```

#### Initialization and Resumption Settings
- `init_from` (str): Determines whether to start training from scratch or resume from a saved model

  - **scratch**: Start training from scratch
  - **resume**: Resume training from a saved model

- `resume_src` (str): Determines the checkpoint to resume training from when init_from is set to '**resume**'

  - **train**: Resume from the last training checkpoint
  - **eval**: Resume from the best evaluation checkpoint
  
#### Training Summary
- Trained a 6.8M param model on 287M examples, which is only ~2% of the dataset
- Trained for 20 hours on a single A100
- final ELO rating: 2118
