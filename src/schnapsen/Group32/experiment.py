"""
Runs all experiments used in Section ... of the paper.
"""
from random import Random
from prepare_ml_models import iterative_training
from result_evaluations import plot_learning_curve
from schnapsen.game import Bot
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.bots.rand import RandBot
from schnapsen.bots.rdeep import RdeepBot
from schnapsen.bots.bully_bot import BullyBot
import pickle
import os

def save_or_load_file(save_file: bool,load_file: bool,LR_single_models_by_opponent=None,LR_mixed_models=None,NN_single_models_by_opponent=None,NN_mixed_models=None):
    """
    Helper function to use when saving/loading the models after trainig to not waste time generating them everytime when wanting to run evaluations

    Args:
        save_file: bool - True|False if we want the models to be saved
        load_file: bool - True|False if we want the models to be loaded
        sending dict/list with save_file=True will save them
    
    Return:
        If load_file=True returns previously saved dict/list
    """
    if save_file and load_file:
        raise ValueError("Choose either save_file OR load_file, not both.")

    if save_file:
        with open("lr_results.pkl", "wb") as f:
            pickle.dump(
                {
                    "LR_single_models_by_opponent": LR_single_models_by_opponent,
                    "LR_mixed_models": LR_mixed_models,
                    "NN_single_models_by_opponent": NN_single_models_by_opponent,
                    "NN_mixed_models": NN_mixed_models,
                },
                f,
            )

    if load_file:
        if not os.path.exists("lr_results.pkl"):
            raise FileNotFoundError("lr_results.pkl not found. Run training first.")

        with open("lr_results.pkl", "rb") as f:
            data = pickle.load(f)

        LR_single_models_by_opponent = data["LR_single_models_by_opponent"]
        LR_mixed_models = data["LR_mixed_models"]
        NN_single_models_by_opponent = data["NN_single_models_by_opponent"]
        NN_mixed_models = data["NN_mixed_models"]

    return LR_single_models_by_opponent,LR_mixed_models,NN_single_models_by_opponent,NN_mixed_models

experiment_rng = Random(20) # Seed ran in the experiment for stable reproduction of results in the future
training_opponents = [
    RandBot(experiment_rng, 'RandBot'), # Low quality data
    BullyBot(experiment_rng, 'BullyBot'), # Medium Quality data
    RdeepBot(2, 3, experiment_rng, 'RdeepBot_depth3'), #High quality data
    RdeepBot(2, 6, experiment_rng, 'RdeepBot_depth6'), #High+ quality data
    ]    
total_games = 2000 # Experiment quantity
iterations = 50 # Wanted iteration amount

LR_single_models_by_opponent, LR_mixed_models  = iterative_training(training_opponents,total_games,experiment_rng,"LR",iterations)
NN_single_models_by_opponent, NN_mixed_models  = iterative_training(training_opponents,total_games,experiment_rng,"NN",iterations)

save_or_load_file(True,False,LR_single_models_by_opponent,LR_mixed_models,NN_single_models_by_opponent,NN_mixed_models) # Use when saving
LR_single_models_by_opponent,LR_mixed_models,NN_single_models_by_opponent,NN_mixed_models = save_or_load_file(save_file=True,load_file=False) # Use when loading

evaluation_opponent = training_opponents[1] # for now

plot_learning_curve("LR",evaluation_opponent,LR_single_models_by_opponent,LR_mixed_models,total_games,iterations)
plot_learning_curve("NN",evaluation_opponent,NN_single_models_by_opponent,NN_mixed_models,total_games,iterations)

