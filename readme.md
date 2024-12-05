# Searchless Chess with Categorical Gaussian Distribution Value Prediction

Pytorch implementation of the papers [Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494) and [Stop Regressing: Training Value Functions via
Classification for Scalable Deep RL](https://arxiv.org/pdf/2403.03950). A chess model is trained to predict the action value of a given board state and action by converting the value target to a Gaussian distribution and using categorical cross-entropy loss.


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

After the download is complete, the data directory should have the following structure:
```
data/
├── train/
|  ├── action_value-00000-of-02148_data.bag
|  ...
|  └── action_value-xxxxx-of-xxxxx_data.bag
├── test/
|  ├── action_value-00000-of-02148_data.bag
|  ...
|  └── action_value-xxxxx-of-xxxxx_data.bag
└── download.sh
```

## Training:

To train the model, run the following command:

```bash
python train.py
```

### Training Configuration Variables
This section describes the configuration variables used in the training script which are found in at the beginning of the `train.py` script.

#### Initialization and Resumption Settings
- `init_from` (str): Determines whether to start training from scratch or resume from a saved model.

  - **scratch**: Start training from scratch.
  - **resume**: Resume training from a saved model.

- `resume_src` (str): Determines the checkpoint to resume training from when init_from is set to '**resume**'.

  - **train**: Resume from the last training checkpoint.
  - **eval**: Resume from the best evaluation checkpoint.
  
#### Model Configuration
- `additional_token_registers` (int): Additional tokens that will be added to the model input.

#### Training Parameters
- `train_save_interval` (int): Interval (in batch iterations) at which the training checkpoint is saved.
- `eval_interval` (int): Interval (in batch iterations) at which the model is evaluated during training.
- `num_epochs` (int): Number of epochs for training.
- `batch_size` (int): Number of samples per batch.
  
#### Learning Rate and Optimization
- `bipe_scale` (float): Batch iterations per epoch scale. Can be used to adjust the learning rate schedule.
- `warmup_steps_ratio` (float): Ratio of warmup iterations to the first epoch.
- `start_lr` (float): Initial learning rate during warmup.
- `max_lr` (float): Maximum learning rate at the end of warmup.
- `final_lr` (float): Final learning rate of the cosine annealing schedule.
- `grad_clip` (float): Gradient clipping value to prevent exploding gradients.

#### Miscellaneous
- `random_seed` (int): Random seed for reproducibility.
- `dataloader_workers` (int): Number of workers for the train and eval dataloaders.
  
#### Training Summary
- Trained a 6M param version of this model on 5.12M examples.
  - loss: 4.7 (100 batches), 4.0 (10000 batches), 3.83 (20000 batches) 3.79 (30000 batches)
  - used ~ 200 MB out of 2 TB training data, or 0.01%
  - training on a potato of a GPU (GTX 1060)

#### Future Work
- use cloud compute to scale up training data
- perform scaling analysis to roughly project peak performance
- train a smaller (1M param) and larger (36M param) one
- log-log plots
