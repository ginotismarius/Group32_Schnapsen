import pickle
from schnapsen.game import Bot
from schnapsen.bots import MLPlayingBot, train_ML_model
import pandas as pd
from pathlib import Path
from random import Random

def updating_ml_bot(opponent_bot: Bot, num_games: int,replay_memory_dir: Path):
    """

        opponent_bot (Bot): The opponent bot.
        num_games (int): Number of games to play for data collection.
    """
    training_data = {
    "opponent": opponent_bot.__class__.__name__,
    "num_games": num_games,
    "data": []
    }
    # Implement the logic to play games and collect training data
    ml_bot = MLPlayingBot()
    replay_memory_dir.mkdir(parents=True, exist_ok=True)

    for game_index in range(num_games):
        game_data = play_one_game(ml_bot, opponent_bot, game_index)
        training_data["data"].append(game_data)

        # Retrain the ML bot every 5 games
        if (game_index + 1) % 5 == 0:
            replay_file = replay_memory_dir / "replay_memory.pkl"

            with open(replay_file, 'wb') as f:
                pickle.dump(training_data["data"], f)

            ml_bot = train_ML_model(replay_memory_location= replay_file,model_location=None)

    return training_data
    
def play_one_game(ml_bot: MLPlayingBot, opponent_bot: Bot, game_index: int):
    """
    Play a single game between the ML bot and the opponent bot.

    Args:
        ml_bot (MLPlayingBot): The machine learning bot.
        opponent_bot (Bot): The opponent bot.
        game_index (int): The index of the current game.
    """
    return {
        "game_index": game_index,
        "ml_bot_moves": [],
        "opponent_bot_moves": [],
        "outcome": "win"  # or "lose" or "draw"
    }