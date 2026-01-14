

"""

Create functions to generate training data for the Schnapsen game. with different amount of games and different opponents.

Save the generated data to files for later use in training. Save files as: ML_bot_<opponent>_<number_of_games>.data
%%time
train_ML_model(Path('ML_replay_memories') / 'bully_replay_memory', Path('ML_models') / 'bully_model', "LR")
"""


from schnapsen.game import Bot, Move, PlayerPerspective
from schnapsen.game import SchnapsenTrickScorer
from schnapsen.deck import Card, Suit, Rank
from schnapsen.bots import MLPlayingBot, MLDataBot, BullyBot, RdeepBot, train_ML_model
import pandas as pd
from pathlib import Path
from random import Random
from pathlib import Path
import pickle


def generate_single_training_data(opponent_bot: Bot, num_games: int):
    """
    Generate training data by playing a specified number of games against a given opponent bot.
    
    Args:
        opponent_bot (Bot): The bot to play against.
        num_games (int): The number of games to play.
    """
    training_data = []
    file_name = f"ML_bot_{opponent_bot.__class__.__name__}_{num_games}.data"

    Path("ML_training_data").mkdir(parents=True, exist_ok=True)
    Path("ML_models").mkdir(parents=True, exist_ok=True)


    ml_data = MLDataBot(bully, Path('ML_replay_memories') / 'bully_replay_memory')
    
    for _ in range(num_games):
        # Initialize game and bots
        # Play game and collect data
        # Append collected data to training_data list
        pass  # Placeholder for game logic
    
    # Save training data to file
    with open(file_name, 'wb') as f:
        pickle.dump(training_data, f)

def generate_mixed_training_data(opponent_bot_list: list, num_games: int, file_name: str):
    """
    Generate training data by playing a specified number of games against a given opponent bot.
    
    Args:
        opponent_bot_list (list): The list of bots to play against.
        num_games (int): The number of games to play.
        file_name (str): The name of the file to save the training data.
    """
    training_data = []
    
    for _ in range(num_games):
        # Initialize game and bots
        # Play game and collect data
        # Append collected data to training_data list
        pass  # Placeholder for game logic
    
    # Save training data to file
    with open(file_name, 'wb') as f:
        pickle.dump(training_data, f)

bully = BullyBot(Random())
rdeep = RdeepBot(10, 5, Random())
generate_single_training_data(bully, 10)
generate_single_training_data(rdeep, 10)