from schnapsen.game import Bot, SchnapsenGamePlayEngine
from schnapsen.bots import MLPlayingBot, BullyBot, RdeepBot, RandBot
from random import Random
from pathlib import Path
import math
import re
import matplotlib.pyplot as plt


#standing later use tournament?

def evaluate_model_training_performance(test_bot: Bot, opponent: Bot, total_games: int=200) -> float:
    """Evaluate the performance of a trained ML bot against a given opponent.
    Args:
        test_bot (Bot): The trained ML bot to be evaluated.
        opponent_bot (Bot): The bot to play against.
        total_games (int): Number of games to play for evaluation.
    Returns:
        float: Win rate of the trained ML bot.
    """
    rng = Random() # Fixed test seed for reproducibility
    eng = SchnapsenGamePlayEngine()
    wins = 0
    test_bot_poits = []
    for _ in range(total_games):
        winner, game_points, _ = eng.play_game(test_bot, opponent, rng)
        if str(winner) != str(opponent):
            test_bot_poits.append(game_points)
            wins += 1

    return wins / total_games


def exctract_learning_curve(model_dir: Path, opponent: Bot, total_games: int, model_class: str):
    """Evaluate the learning curve of a trained ML bot against a given opponent.
    Args:
        model_dir (Path): Directory where the trained model is located.
        opponent (Bot): The bot to play against.
        total_games (int): Total number of games the model was trained on.
        model_class (str): The class/type of the ML model (e.g., 'NN', 'LR').
    Returns:
        (x,y): A tuple containing lists of number of games played and corresponding win rates.
    """
    points = []

    opponent_name = str(opponent)

    for model_file in model_dir.glob(f"*{opponent_name}*{model_class}*.model"):
        if "final" in model_file.name:
            iteration = float("inf")
        else:
            match = re.search(r"iteration_(\d+)", model_file.name)
            iteration = int(match.group(1))

        ml_bot = MLPlayingBot(model_location=model_file)
        print(f"Evaluating {model_file.name}")
        win_rate = evaluate_model_training_performance(ml_bot, opponent)
        points.append((iteration, win_rate))

    points.sort(key=lambda p: p[0])

    x = []
    y = []
    step = total_games // (len(points) - 1)
    games_seen = 0
    for iteration, win_rate in points:
        x.append(total_games if iteration == float("inf") else games_seen)
        y.append(win_rate)
        games_seen += step

    return x, y

def plot_learning_curve(nn_model_dir: Path, lr_model_dir: Path, training_opponent: Bot, opponent: Bot, total_games: int):
    """Plot the learning curves of NN and LR models against a given opponent.
    Args:
        nn_model_dir (Path): Directory where the NN trained models are located.
        lr_model_dir (Path): Directory where the LR trained models are located.
        training_opponent (Bot): The bot used for training.
        opponent (Bot): The bot to play against.
        total_games (int): Total number of games the models were trained on.
    """
    plt.figure(figsize=(12, 8))
    nn_x, nn_y = exctract_learning_curve(nn_model_dir, opponent, total_games, "NN")
    lr_x, lr_y = exctract_learning_curve(lr_model_dir, opponent, total_games, "LR")

    plt.plot(nn_x, nn_y, label="Neural Network", marker='o')
    plt.plot(lr_x, lr_y, label="Logistic Regression", marker='x')
    plt.xlabel("Number of Training Games")
    plt.ylabel("Win Rate")
    plt.title(f"Learning Curve trained against {training_opponent} vs {opponent}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, total_games)
    plt.show()
    plt.savefig(f"learning_curve_vs_{training_opponent}_{total_games}_games.png")
    plt.close()