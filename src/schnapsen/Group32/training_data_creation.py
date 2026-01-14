"""

Create functions to generate training data for the Schnapsen game. with different amount of games and different opponents.

Save the generated data to files for later use in training. Save files as: ML_bot_<opponent>_<number_of_games>.data

"""

from schnapsen.game import Bot
from schnapsen.bots import MLPlayingBot, MLDataBot, BullyBot, RdeepBot, train_ML_model
import pandas as pd
from pathlib import Path
from random import Random
from train_bot import updating_ml_bot
import pickle


def generate_single_training_data(opponent_bot: Bot, num_games: int):
    """
    Generate training data by playing a specified number of games against a given opponent bot.
    
    Args:
        opponent_bot (Bot): The bot to play against.
        num_games (int): The number of games to play.
    """

    # Determine file paths
    current_dir = Path(__file__).resolve().parent
    current_root = current_dir.parent

    final_training_data_dir = current_root / "ML_training_data"
    final_training_data_dir.mkdir(parents=True, exist_ok=True)

    replay_memory_dir = current_root / "ML_replay_memories"
    replay_memory_dir.mkdir(parents=True, exist_ok=True)

    model_dir = current_root / "ML_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    file_path = final_training_data_dir / f"ML_bot_{opponent_bot.__class__.__name__}_{num_games}.data"

    # Collect training data
    #training_data = updating_ml_bot(opponent_bot, num_games, replay_memory_dir)
    training_data = {}
    # Save final training data to file
    with open(file_path, 'wb') as f:
        pickle.dump(training_data,f)

    #show The final generated data
    #df = pd.DataFrame(training_data["data"])
    #print(df)

bully = BullyBot(Random())
rdeep = RdeepBot(10, 5, Random())
generate_single_training_data(bully, 10)
generate_single_training_data(rdeep, 10)