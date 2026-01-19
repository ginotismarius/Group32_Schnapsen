"""

Create functions to generate training data for the Schnapsen game. with different amount of games and different opponents.

Save the generated data to files for later use in training. Save files as: ML_bot_<opponent>_<number_of_games>.data

"""
from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MiniMaxBot, AlphaBetaBot, BullyBot, RdeepBot, RandBot
from pathlib import Path
from random import Random
from train_bot import updating_ml_bot

#import pandas as pd



def generate_single_training_data(opponent_bot: Bot, num_games: int):
    """
    Generate training data by playing a specified number of games against a given opponent bot.
    
    Args:
        opponent_bot (Bot): The bot to play against.
        num_games (int): The number of games to play.
    """

    # Determine file paths
    root_dir = Path(__file__).resolve().parent

    replay_dir = root_dir / "ML_replay_memories"
    replay_dir.mkdir(parents=True, exist_ok=True)

    model_dir = root_dir / "ML_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training data against {opponent_bot} for {num_games} games.")

    model_path = updating_ml_bot(opponent_bot, num_games, replay_dir,model_dir)

    if model_path is not None:
        raise RuntimeError("Model training failed, no model path returned.")
    
    print(f"Training data saved to {model_path}\n")


# Define different opponents
set_rng = Random(12345)
opponents = [
    BullyBot(set_rng, 'BullyBot'),
    RdeepBot(10, 5, set_rng, 'RdeepBot_depth5'),
    #AlphaBetaBot('AlphaBetaBot'), Bot only works in phase 2, use another bot in phase 1 then switch somehow ?
    #MiniMaxBot('MiniMaxBot'),
    RandBot(set_rng, 'Randy'),
]

# Define different numbers of games
game_counts = [5, 10, 50]

for opponent in opponents:
    for num_games in game_counts:
        generate_single_training_data(opponent, num_games)