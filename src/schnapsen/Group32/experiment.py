"""
File for running all experiments for our research question. Used to test our hypothesis
"""
from random import Random
from prepare_ml_models import iterative_training
from result_evaluations import plot_learning_curve, plot_model_heatmap, plot_generalization_curve
from schnapsen.bots import RandBot, BullyBot, RdeepBot
import pickle
import os

def save_file(
        filepath="results.pkl",
        LR_single_models_by_opponent=None,
        LR_mixed_models=None,
        NN_single_models_by_opponent=None,
        NN_mixed_models=None,
    ):
    """
    Save trained models to disk.
    """
    with open(filepath, "wb") as f:
        pickle.dump(
            {
                "LR_single_models_by_opponent": LR_single_models_by_opponent,
                "LR_mixed_models": LR_mixed_models,
                "NN_single_models_by_opponent": NN_single_models_by_opponent,
                "NN_mixed_models": NN_mixed_models,
            },
            f,
        )

def load_file(filepath="results.pkl"):
    """
    Load trained models from save.

    Returns:
        LR_single_models_by_opponent,
        LR_mixed_models,
        NN_single_models_by_opponent,
        NN_mixed_models
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run training first.")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return (
        data["LR_single_models_by_opponent"],
        data["LR_mixed_models"],
        data["NN_single_models_by_opponent"],
        data["NN_mixed_models"],
    )

def collect_final_models(single_models_by_opponent, mixed_models):
    """
    Extract the final trained model for each training approach.

    Args:
        single_models_by_opponent: dict[Bot, list[Bot]]
        mixed_models: list[Bot]

    Returns:
        list[Bot]: Final models only
    """
    final_models = []

    for _, models in single_models_by_opponent.items():
        final_models.append(models[-1])  # last iteration

    final_models.append(mixed_models[-1])  # final mixed model

    return final_models

experiment_rng = Random(20) # Seed ran in the experiment for stable reproduction of results in the future
training_opponents = [ # Experiment quality
    RandBot(experiment_rng, 'RandBot'), # Low quality data
    BullyBot(experiment_rng, 'BullyBot'), # Medium Quality data
    RdeepBot(2, 3, experiment_rng, 'RdeepBot_depth3'), #High quality data
    RdeepBot(3, 5, experiment_rng, 'RdeepBot_depth5'), #High quality data
    ]    
total_games_lr = 5000 # LR model class experiment quantity
total_games_nn = 30000 # NN model class experiment quantity
iterations_lr = 50 # LR model class experiment iteration amount
iterations_nn = 100 # NN model class experiment iteration amount
evaluation_games = 250 # Number of games per evaluation run (more games = lower variance in win rate estimate | smaller std)
evaluation_repeats = 20 # Number of independent evaluation runs with different seeds (more repeats = smoother and more reliable mean)

""" # Run when training new models
LR_single_models_by_opponent, LR_mixed_models  = iterative_training(training_opponents,total_games_lr,experiment_rng,"LR",iterations_lr)
NN_single_models_by_opponent, NN_mixed_models  = iterative_training(training_opponents,total_games_nn,experiment_rng,"NN",iterations_nn)
"""

""" # Saving/Loading models after they have been trained (Use 1 save or load not both)
save_file(LR_single_models_by_opponent,LR_mixed_models,NN_single_models_by_opponent,NN_mixed_models) # Use to save models to skip training in reruns
LR_single_models_by_opponent,LR_mixed_models,NN_single_models_by_opponent,NN_mixed_models = load_file() # Use to load previously saved models
"""

"""# Learning curves vs 1 opponent
evaluation_opponent_rand = training_opponents[0] # for now
evaluation_opponent_bully = training_opponents[1] # for now
evaluation_opponent_rdeep3 = training_opponents[2] # for now

plot_learning_curve("LR",evaluation_opponent_rand,LR_single_models_by_opponent,LR_mixed_models,total_games_lr,iterations_lr,2,evaluation_games,evaluation_repeats) # vs Randy
plot_learning_curve("NN",evaluation_opponent_rand,NN_single_models_by_opponent,NN_mixed_models,total_games_nn,iterations_nn,4,evaluation_games,evaluation_repeats)

plot_learning_curve("LR",evaluation_opponent_bully,LR_single_models_by_opponent,LR_mixed_models,total_games_lr,iterations_lr,2,evaluation_games,evaluation_repeats) # vs Bully
plot_learning_curve("NN",evaluation_opponent_bully,NN_single_models_by_opponent,NN_mixed_models,total_games_nn,iterations_nn,4,evaluation_games,evaluation_repeats)

plot_learning_curve("LR",evaluation_opponent_rdeep3,LR_single_models_by_opponent,LR_mixed_models,total_games_lr,iterations_lr,2,evaluation_games,evaluation_repeats) # vs Rdeep3
plot_learning_curve("NN",evaluation_opponent_rdeep3,NN_single_models_by_opponent,NN_mixed_models,total_games_nn,iterations_nn,4,evaluation_games,evaluation_repeats)
"""

"""# Learning curve mean
all_evaluation_opponents = [training_opponents[0],training_opponents[1],training_opponents[2]]

plot_generalization_curve("LR",all_evaluation_opponents,LR_single_models_by_opponent,LR_mixed_models,total_games_lr,iterations_lr,3,evaluation_games,evaluation_repeats)
plot_generalization_curve("NN",all_evaluation_opponents,NN_single_models_by_opponent,NN_mixed_models,total_games_nn,iterations_nn,5,evaluation_games,evaluation_repeats)
"""

"""# Heat maps
LR_final_models = collect_final_models(LR_single_models_by_opponent,LR_mixed_models)
NN_final_models = collect_final_models(NN_single_models_by_opponent,NN_mixed_models)
final_models = LR_final_models + NN_final_models

plot_model_heatmap(LR_final_models,"LR Models Heatmap")
plot_model_heatmap(NN_final_models,"NN Models Heatmap")
plot_model_heatmap(final_models,"All Models Heatmap")
"""