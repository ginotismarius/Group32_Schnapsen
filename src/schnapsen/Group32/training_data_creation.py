from turtle import pd
from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MiniMaxBot, AlphaBetaBot, BullyBot, RdeepBot, RandBot , MLDataBot, MLPlayingBot
from pathlib import Path
from random import Random
from train_bot import updating_ml_bot
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

def single_iterative_training(opponent_bot: Bot, total_games: int, model_class: str,iterations: int =3):
    """Generate training data by updating the ML bot iteratively against a given opponent.
    Args:
        opponent_bot (Bot): The opponent bot to play against.
        total_games (int): The total number of games to play for training data.
        model_training_class (str): The class/type of the ML model (e.g., 'NN', 'LR').
        iterations (int): Number of iterations to update the ML bot.
    """

    games_per_iteration = total_games // iterations

    replay_dir = ROOT_DIR / f"{model_class}_ML_replay_memories"
    model_dir = ROOT_DIR / f"{model_class}_ML_models"
    behaviour_bot = RandBot(Random(10)) # Using a random bot as the base behaviour bot

    for iteration in range(iterations):
        replay_file = replay_dir / f"replay_memory_{opponent_bot}_{total_games}_{model_class}_iteration_{iteration}.txt"
        model_path = model_dir / f"ML_bot_{opponent_bot}_{total_games}_{model_class}_iteration_{iteration}.model"
        test_file = updating_ml_bot(behaviour_bot, opponent_bot, games_per_iteration, replay_file, model_path, model_class)
        if test_file is not None:
            behaviour_bot = MLPlayingBot(model_location=model_path)
        else:
            print(f"Skipping iteration {iteration} due to insufficient data.")


    print(
        f"Finished {model_class} training vs {opponent_bot} "
        f"({total_games} games, {iterations} iterations)"
    )

# Define different opponents
set_rng = Random()
opponents = [
    BullyBot(set_rng, 'BullyBot'),
    RdeepBot(5, 5, set_rng, 'RdeepBot_depth5'),
    RdeepBot(5, 10, set_rng, 'RdeepBot_depth10'),
    RandBot(set_rng, 'Randy'),
]

# Define different numbers of games, and training modes
game_counts = [30,60,90]
training_modes = ['NN', 'LR']

for opponent in opponents:
    for total_games in game_counts:
        for model_class in training_modes:
            single_iterative_training(opponent, total_games, model_class)