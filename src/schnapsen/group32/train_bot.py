import pickle
from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MLPlayingBot, train_ML_model, MLDataBot,RandBot
import pandas as pd
from pathlib import Path
from random import Random

def updating_ml_bot(behaviour_ml_bot: Bot,opponent_bot: Bot, num_games: int,replay_file: Path, model_path: Path, model_class: str) -> Path:
    """
    Update the ML bot by playing a specified number of games against a given opponent bot,
    saving the replay memory and trained model to specified directories.
    Args:
        behaviour_ml_bot (Bot): The ML bot to be updated.
        opponent_bot (Bot): The bot to play against.
        num_games (int): The number of games to play.
        replay_memory_dir (Path): Directory to save replay memory files.
        model_dir (Path): Directory to save trained model files.
        model_class (str): The class/type of the ML model (e.g., 'NN', 'LR').
    """
    eng = SchnapsenGamePlayEngine()

    replay_file.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    ml_bot = MLDataBot(bot=behaviour_ml_bot, replay_memory_location=replay_file)

    for game_index in range(num_games):
        eng.play_game(ml_bot,opponent_bot,Random(game_index+42)) # Fixed seed for reproducibility 
    # After playing the games, train the ML model
    if check_replay_file(replay_file):
        train_ML_model(replay_memory_location=replay_file, model_location=model_path, model_class=model_class)
        return model_path
    else:
        print(f"Insufficient data in replay file {replay_file} vs {opponent_bot}, skipping model training.")
        #print("ML only lost or won all games, need at least one of each to train.")
        return None





def check_replay_file(replay_file: Path) -> bool:
    if not replay_file.exists():
        return False

    labels = set()
    with open(replay_file, "r") as f:
        for line in f:
            if "||" not in line:
                continue
            _, label = line.strip().split("||")
            labels.add(label)
            if len(labels) >= 2:
                return True
    return False