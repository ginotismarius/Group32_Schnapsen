from schnapsen.game import Bot, SchnapsenGamePlayEngine
from schnapsen.bots import MLPlayingBot, BullyBot, RdeepBot, RandBot
from random import Random
from pathlib import Path
import math
import re
import matplotlib.pyplot as plt
import numpy as np

#standing later use tournament?

def evaluate_model_training_performance(model_path: Path, opponent: Bot, eval_games: int=25,repeats: int =5) -> float:

    eng = SchnapsenGamePlayEngine()
    win_rates = []

    for seed in range(repeats):
        rng = Random(seed)
        wins = 0

        for _ in range(eval_games):
            test = MLPlayingBot(model_location=model_path)
            winner, _, _ = eng.play_game(test, opponent, rng)
            if winner is test:
                wins += 1  
        
        win_rates.append(wins/eval_games)
    
    return np.mean(win_rates), np.std(win_rates)


def exctract_learning_curve(model_dir: Path, opponent: Bot, total_games: int,trained_against: Bot, model_class: str, min_iteration_step):
    """Evaluate the learning curve of a trained ML bot against a given opponent.
    Args:
        
    Returns:
    """
    points = []

    for model_file in model_dir.glob(f"*{trained_against}*{total_games}*{model_class}*.model"):
        if "final" in model_file.name:
            iteration = total_games // min_iteration_step
            games_seen = total_games
        else:
            match = re.search(r"iteration_(\d+)", model_file.name)
            if not match:
                continue
            iteration = int(match.group(1))
            games_seen = iteration * min_iteration_step

        print(f"Evaluating {model_file.name}")

        mean_win,std_win = evaluate_model_training_performance(model_file, opponent)
        points.append((games_seen, mean_win,std_win))

    points.sort(key=lambda p: p[0])

    x = [p[0] for p in points]
    y_mean = [p[1] for p in points]
    y_std = [p[2] for p in points]
    return x, y_mean, y_std

def smooth_curve(y, window_size):
    if len(y) < window_size:
        return np.array(y)
    return np.convolve(y,np.ones(window_size)/window_size,mode="same")

def confidence_intervals(mean,std,z=1.96):
    mean = np.array(mean)
    std = np.array(std)
    return mean - z * std, mean + z * std


def plot_learning_curve(nn_model_dir: Path, lr_model_dir: Path, training_opponent: Bot, opponent: Bot, total_games: int,min_iteration_step: int,smooth_window: int = 2):
    plt.figure(figsize=(12, 8))

    nn_x,nn_mean,nn_std = exctract_learning_curve(nn_model_dir, opponent, total_games,training_opponent,"NN",min_iteration_step)
    lr_x,lr_mean,lr_std = exctract_learning_curve(lr_model_dir, opponent, total_games,training_opponent,"LR",min_iteration_step)

    nn_mean_s = smooth_curve(nn_mean, smooth_window)
    nn_std_s    = smooth_curve(nn_std, smooth_window)

    lr_mean_s = smooth_curve(lr_mean, smooth_window)
    lr_std_s    = smooth_curve(lr_std, smooth_window)

    nn_x_s = nn_x[:len(nn_mean_s)]
    lr_x_s = lr_x[:len(lr_mean_s)]

    nn_low,nn_high = confidence_intervals(nn_mean_s,nn_std_s)
    lr_low,lr_high = confidence_intervals(lr_mean_s,lr_std_s)


    plt.plot(nn_x_s, nn_mean_s, label="Neural Network", marker='o')
    plt.fill_between(nn_x_s, nn_low, nn_high, alpha=0.2)

    plt.plot(lr_x_s, lr_mean_s, label="Logistic Regression", marker='x')
    plt.fill_between(lr_x_s, lr_low, lr_high, alpha=0.2)

    plt.xlabel("Number of Training Games")
    plt.ylabel("Win Rate")
    plt.title(f"Learning Curve trained against {training_opponent} Evaluated by playing against: {opponent}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, total_games)
    plt.tight_layout()
    plt.savefig(f"Learning_curve_trained_against_{training_opponent}_evaluated__against_{opponent}_{total_games}_.png")
