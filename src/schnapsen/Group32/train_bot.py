import pickle
from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MLPlayingBot, train_ML_model, MLDataBot,RandBot
import pandas as pd
from pathlib import Path
from random import Random

def updating_ml_bot(opponent_bot: Bot, num_games: int,replay_memory_dir: Path, model_dir: Path = None) -> Path:
    """

        opponent_bot (Bot): The opponent bot.
        num_games (int): Number of games to play for data collection.
    """
    eng = SchnapsenGamePlayEngine()

    replay_memory_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    replay_file = replay_memory_dir / f"replay_memory_{opponent_bot.__class__.__name__}_{num_games}_NN.txt"
    model_path = model_dir / f"ML_bot_{opponent_bot.__class__.__name__}_{num_games}_NN.model"

    # Making the ML that has no data so plays randomly at first setting first seed for reproducibility to 10
    ml_bot = MLDataBot(bot=RandBot(Random(10)), replay_memory_location=replay_file)
    trained_model_path = None
    

    for game_index in range(num_games):
        eng.play_game(ml_bot,opponent_bot,Random())
        # Retrain the ML bot every 5 games ? (Broken since overwrite is not implemented by in base code ) but the ML bot keeps learning from the replay memory anyway every game | used MLDAtaBot over MLPlayingBot to allow continuous learning
        """
        if (game_index + 1) % 5 == 0:
            trained_model_path = train_ML_model(replay_memory_location=replay_file, model_location=model_path, model_class='NN')  
        """
    if trained_model_path is None:
        trained_model_path = train_ML_model(replay_memory_location=replay_file, model_location=model_path, model_class='NN')
    return trained_model_path

    # Load and return the final training data

    def load_trained_ml_bot(model_path: Path) -> MLPlayingBot:
        """Load a trained MLPlayingBot from the specified model path."""
        return MLPlayingBot(model_location=model_path)