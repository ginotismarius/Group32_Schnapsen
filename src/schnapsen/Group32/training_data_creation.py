from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MiniMaxBot, AlphaBetaBot, BullyBot, RdeepBot, RandBot , MLDataBot, MLPlayingBot
from pathlib import Path
from random import Random
from train_bot import updating_ml_bot
from learning_evaluation import plot_learning_curve
from pathlib import Path

FINAL_MODEL_LIST = []
ROOT_DIR = Path(__file__).resolve().parent
MIN_ITERATION_STEP = {
    100: 5,
    200: 20,
    500: 25,
    1000: 50,
    2500: 100,
    10000: 100,
    10001: 250,
    10002: 500,
    10005: 1000
    }

def single_iterative_training(opponent_bot: Bot, total_games: int, model_class: str):

    min_iteration_step = MIN_ITERATION_STEP.get(total_games, total_games // 10)
    iterations = max(total_games // min_iteration_step, 1)
    games_per_iteration = total_games // iterations

    replay_dir = ROOT_DIR / f"{model_class}_ML_replay_memories"
    model_dir = ROOT_DIR / f"{model_class}_ML_models"
    replay_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    replay_file = replay_dir / f"replay_memory_{opponent_bot}_{total_games}_{model_class}.txt"
    if replay_file.exists():
        replay_file.unlink()

    behaviour_bot = RandBot(Random(10)) # Using a random bot as the base behaviour bot

    for iteration in range(iterations):
        model_path = model_dir / f"ML_bot_{opponent_bot}_{total_games}_{model_class}_iteration_{iteration}.model"

        # Try to update the ML bot to learn from the games it just played after playing games_per_iteration games
        trained_model = updating_ml_bot(behaviour_bot, opponent_bot, games_per_iteration, replay_file, model_path, model_class,total_games)
        if trained_model is not None:
            behaviour_bot = MLPlayingBot(model_location=model_path)
        else:
            print(f"Skipping iteration {iteration} due to insufficient data.")

    # Save the final model
    final_model_path = model_dir / f"ML_bot_{opponent_bot}_{total_games}_{model_class}_final.model"
    updating_ml_bot(behaviour_bot, opponent_bot, games_per_iteration, replay_file, final_model_path, model_class,total_games)
    final_bot = MLPlayingBot(model_location=final_model_path,name=f"Final_{model_class}_vs_{opponent_bot}_{total_games}")
    FINAL_MODEL_LIST.append(final_bot)


def create_trained_singletype_models(opponents,game_counts,training_modes):
    for opponent in opponents:
      for total_games in game_counts:
           for model_class in training_modes:
               single_iterative_training(opponent, total_games, model_class)
    print("Final trained list:")
    count = 0
    for final_model in FINAL_MODEL_LIST:
        count += 1
        print(f"Model {count}: ", final_model)
