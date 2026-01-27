Schnapsen ML Experiments
=======================

This repository contains code for running machine learning experiments in the game of Schnapsen.
The goal is to evaluate how training data quality (opponent strength) and training strategy
(single opponent vs mixed opponents) affect performance and generalization of ML agents.

The experiments compare:
- Logistic Regression (LR) models
- Neural Network (NN) models
- Single-opponent training vs mixed-opponent training

Structure
========================

experiment.py
    Main entry point for running experiments, training models,
    loading/saving results, and generating plots.

prepare_ml_models.py
    Contains the training logic for ML agents, including:
    - Iterative training
    - Single-opponent training
    - Mixed-opponent training
    - Replay memory handling

result_evaluation.py
    Functions for evaluating trained models and producing:
    - Learning curves
    - Generalization curves
    - Win-rate heatmaps

========================

Requirements
========================
- Python 3.10 or newer (Worked on 3.13.9)
- All other required Python packages are listed in requirements.txt

Install dependencies: (GitHub only)
    by running setup_pyvenv.bat 
    or
    pip install -r requirements.txt
========================

How to run the experiment:

All experiments are controlled from experiment.py.

Training ML models can take a significant amount of time. For convenience results.zip is included and contains pre-trained
models used in the evaluation.To save time, it is recommended to load these models instead of retraining.

- All experiments use a fixed random seed: experiment_rng = Random(20)
- Evaluation runs use multiple seeds to reduce variance
- Results should be reproducible across runs
========================

Step 0. (Optional)
Select training parameters in experiment.py with wanted values. These will be used when training the models.
These parameters control training and evaluation behaviour. Comments in the code explain each one.

Key parameters:

training_opponents
total_games_lr
total_games_nn
iterations_lr
iterations_nn
evaluation_games
evaluation_repeats
========================

Step 1. Training New Models (Should be skipped to when loading from file): (GitHub only)
Remove """ from line 89|92. 
The code inside this block will Train LR and NN models with selected parameters.
Save iterations of the models and their replays.
========================

Step 2. Saving/Loading the ML models.

Step 2.1. Saving trained models

Remove """ from line 94|97.
Remove # from line 95.
The code section saves the models so training can be skipped in future runs.

!!!  Recommended to start from this step! (GitHub only)
Step 2.2. Loading the models from previous save. (Highly recommended)

Remove """ from line 94|97.

Note: Extract "results.zip" and ensure file is located in the same directory as experiment.py. (Other locations untested)
Currently saved models in results.zip are: (GitHub only)
LR Models:
* Trained on 5000 games
* 50 training iterations
NN Models:
* Trained on 30000 games
* 100 training iterations
Shared:
* Trained against:
RandBot, BullyBot, Rdeep_3, Rdeep_5, Mixed
========================

Step 3. Select which experiments to run. (Can be one or all, but the time for each additional experiment will add up)

Note: for basic testing it is recommended to lower evaluation_games and evaluation_repeats.

Step 3.1. Learning curves (Single opponent)

Remove """ from line 99|112
Select wanted opponent for evaluation against trained ML models
currently options made for Randbot,BullyBot,Rdeep(Depth 3).

(Optional) - Making your own opponent bot
evaluation_opponent_NewBot = Assign a bot of choice.


To evaluate against a different opponent, change the second argument in plot_learning_curve

Step 3.2. Learning Curves (Mean over All Opponents):

Remove """ from line 114|119

Specify all opponents used for evaluation:
all_evaluation_opponents = [Bot1, Bot2, ...]

Step 3.3. Model Heatmaps (Evaluating trained ML models against each other directly)

Remove """" from line 121|129

No additional changes are required.
========================

Step 4. Run the file experiment.py
