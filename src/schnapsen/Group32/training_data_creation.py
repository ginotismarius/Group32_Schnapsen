from turtle import pd
from schnapsen.game import Bot , SchnapsenGamePlayEngine
from schnapsen.bots import MiniMaxBot, AlphaBetaBot, BullyBot, RdeepBot, RandBot , MLDataBot, MLPlayingBot
from pathlib import Path
from random import Random
from train_bot import updating_ml_bot
from learning_evaluation import plot_learning_curve
from pathlib import Path

FINAL_MODEL_LIST = []
ROOT_DIR = Path(__file__).resolve().parent

def single_iterative_training(opponent_bot: Bot, total_games: int, model_class: str,min_iteration_step: int =200):
    """Generate training data by updating the ML bot iteratively against a given opponent.
    Args:
        opponent_bot (Bot): The bot to play against.
        total_games (int): Total number of games to be played for training.
        model_class (str): The class/type of the ML model (e.g., 'NN', 'LR').
        min_iteration_step (int): Minimum number of games per iteration. (how often attempt to update the model)
    """

    iterations = max(total_games // min_iteration_step, 1)
    games_per_iteration = total_games // iterations

    replay_dir = ROOT_DIR / f"{model_class}_ML_replay_memories"
    model_dir = ROOT_DIR / f"{model_class}_ML_models"

    behaviour_bot = RandBot(Random(10)) # Using a random bot as the base behaviour bot

    for iteration in range(iterations):
        replay_file = replay_dir / f"replay_memory_{opponent_bot}_{total_games}_{model_class}_iteration_{iteration}.txt"
        model_path = model_dir / f"ML_bot_{opponent_bot}_{total_games}_{model_class}_iteration_{iteration}.model"
        # Try to update the ML bot to learn from the games it just played after playing games_per_iteration games
        test_file = updating_ml_bot(behaviour_bot, opponent_bot, games_per_iteration, replay_file, model_path, model_class)
        if test_file is not None:
            behaviour_bot = MLPlayingBot(model_location=model_path)
        else:
            print(f"Skipping iteration {iteration} due to insufficient data.")

    # Save the final model
    replay_file = replay_dir / f"replay_memory_{opponent_bot}_{total_games}_{model_class}_final.txt"
    final_model_path = model_dir / f"ML_bot_{opponent_bot}_{total_games}_{model_class}_final.model"
    test_file = updating_ml_bot(behaviour_bot, opponent_bot, total_games, replay_file, final_model_path, model_class)
    if test_file is not None:
        final_bot = MLPlayingBot(model_location=final_model_path, name=f"Final_{model_class}_ML_bot_vs_{opponent_bot}_{total_games}_games")
        FINAL_MODEL_LIST.append(final_bot)
    else:
        final_bot = behaviour_bot  # Use the last successful bot
        FINAL_MODEL_LIST.append(final_bot)
        #print(f"Final model saved at {final_model_path} for opponent {opponent_bot} after {total_games} games.")
    #print(f"Finished {model_class} training vs {opponent_bot} | {total_games} games, {iterations} iterations)")

# Define different opponents
set_rng = Random(20)
opponents = [
    BullyBot(set_rng, 'BullyBot'),
    RdeepBot(2, 5, set_rng, 'RdeepBot_depth5'),
    RdeepBot(4, 10, set_rng, 'RdeepBot_depth10'),
    RandBot(set_rng, 'RandBot'),  
]

# Define different numbers of games, and training modes
game_counts = [10000]
training_modes = ['NN', 'LR']

for opponent in opponents:
    for total_games in game_counts:
        for model_class in training_modes:
            single_iterative_training(opponent, total_games, model_class)


print("Final trained list:")
count = 0
for final_model in FINAL_MODEL_LIST:
    count += 1
    print(f"Model {count}: ", final_model)

for opponent in opponents:
    for total_games in game_counts:
        plot_learning_curve(nn_model_dir=Path("src/schnapsen/Group32/NN_ML_models"),
                    lr_model_dir=Path("src/schnapsen/Group32/LR_ML_models"),
                    training_opponent=opponent,
                    opponent=opponent,
                    total_games=total_games)
        