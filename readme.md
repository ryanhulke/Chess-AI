## Architecture Overview
- The transformer ([Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494)) is trained on a Gaussian distribution over action-values ([Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://arxiv.org/pdf/2403.03950))
- this allows us to frame it as a classification problem, and we can then use cross-entropy loss
- using the Gaussian instead of one-hot encoded discrete action-values stabilizes training


## Setup:

Clone the repository:

```bash
git clone https://github.com/ryanhulke/searchless-chess
cd searchless-chess
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
- `init_from` (str): Determines whether to start training from scratch or resume from a saved model.

  - **scratch**: Start training from scratch.
  - **resume**: Resume training from a saved model.

- `resume_src` (str): Determines the checkpoint to resume training from when init_from is set to '**resume**'.

  - **train**: Resume from the last training checkpoint.
  - **eval**: Resume from the best evaluation checkpoint.
  
#### Training Summary
- Trained a 6.8M param model on 287M examples, which is only ~2% of the dataset
- Trained for 20 hours on a single A100
- final ELO rating: 2118
