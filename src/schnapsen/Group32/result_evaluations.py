from schnapsen.game import Bot, SchnapsenGamePlayEngine
from schnapsen.bots import MLPlayingBot, BullyBot, RdeepBot, RandBot
from random import Random
from pathlib import Path
import math
import re
import matplotlib.pyplot as plt
import numpy as np

def evaluate_learning(bots: list[Bot], opponent: Bot, eval_games: int = 100, repeats: int = 10) -> tuple[list[float], list[float]]:
    """
    Evaluate a sequence of bots representing successive training iterations.

    Args:
        bots: list[Bot] - List of a specific ML model
        opponent: Bot - The opponent we will be testing against
        eval_games: int - How many games we will run to get win rate statistics (default =25)
        repeats: int - Amount of times we will rerun the test with different seeds (default =4)
        used to eliminate "Lucky" games and keep the win rate based on skill.

    Returns:
        means: list[float] - List of the arithmetic mean along the specified axis.
        stds: list[float] - List of the standard deviation along the specified axis.
    """
    means, stds = [],[]

    for bot in bots:
        print(f"Evaluating: {bot}")
        mean, std = evaluate_bot(bot,opponent,eval_games,repeats)
        means.append(mean)
        stds.append(std)

    return means, stds

def evaluate_bot(bot: Bot, opponent: Bot, eval_games: int = 100, repeats: int = 10) -> tuple[float, float]:
    """
    Evaluate a bot by playing repeated games against a fixed opponent.

    Args:
        bot: Bot - ML model we want to test
        opponent: Bot - The opponent we will be testing against
        eval_games: int - How many games we will run to get win rate statistics (default =25)
        repeats: int - Amount of times we will rerun the test with different seeds (default =4)
        used to eliminate "Lucky" games and keep the win rate based on skill.
    
    Returns: 
        mean_win_rate: float - The arithmetic mean along the specified axis.
        std_win_rate: float - The standard deviation along the specified axis.
    """
    
    eng = SchnapsenGamePlayEngine()
    win_rates = []

    for seed in range(repeats):
        rng = Random(seed)
        wins = 0

        for _ in range(eval_games):
            winner, _, _ = eng.play_game(bot, opponent, rng)
            if winner is bot:
                wins += 1  
        
        win_rates.append(wins/eval_games)
    
    return float(np.mean(win_rates)), float(np.std(win_rates))

def smooth_curve(y, window_size):
    if len(y) < window_size:
        return np.array(y)
    return np.convolve(y,np.ones(window_size)/window_size,mode="same")

def confidence_intervals(mean,std,z=1.96):
    mean = np.array(mean)
    std = np.array(std)
    return mean - z * std, mean + z * std

def plot_learning_curve(
        model_class: str,
        evaluate_against: Bot,
        single_models_by_opponent: dict[str,list[Bot]],
        mixed_models: list[Bot],
        total_games: int,
        iterations: int,
        smooth_window: int =1
        ):
    """
    Plot learning curves for single-opponent and mixed-opponent training evaluated against selected opponent

    Args:
        model_class: str - What was used when training the ML model "LR"|"NN"
        evaluate_against: Bot - The opponent we will be matched against when testing
        single_models_by_opponent: dict[str,list[Bot]] - A dictonary where key is what the bot was trained against.
        And values being a list of the Model iterations
        mixed_models: list[Bot] - A list of models trained against mixed oponents with their iterations
        total_games: int - Total games used in training
        iterations: Amount of iterations saved of the models
        smooth_window: int - ...
    """
    plt.figure(figsize=(12, 8))
    games_per_iteration = total_games // iterations
    baseline_bot = RandBot(Random(0), "Baseline")
    baseline_mean, baseline_std = evaluate_bot(baseline_bot, evaluate_against)


    x = [0] + [min((i + 1) * games_per_iteration, total_games) for i in range(len(mixed_models))]

    # Single opponent learning curves
    for opponent_name, bots in single_models_by_opponent.items():
        means, stds = evaluate_learning(bots,evaluate_against)
        means = [baseline_mean] + means
        stds = [baseline_std] + stds

        marker_every = max(1, len(x) // 12)
        means_s = smooth_curve(means,smooth_window)
        stds_s = smooth_curve(stds,smooth_window)
        lower, upper = confidence_intervals(means_s, stds_s)

        plt.plot(x, means_s, linestyle="solid", linewidth=3, label=f"Trained vs {opponent_name}")
        plt.fill_between(x, lower, upper, alpha=0.03)

    # Mixed opponent learning curves
    mixed_means, mixed_stds = evaluate_learning(mixed_models,evaluate_against)
    mixed_means = [baseline_mean] + mixed_means
    mixed_stds = [baseline_std] + mixed_stds

    
    mixed_means_s =smooth_curve(mixed_means,smooth_window)
    mixed_stds_s = smooth_curve(mixed_stds,smooth_window)
    lower, upper = confidence_intervals(mixed_means_s, mixed_stds_s)

    marker_every = max(1, len(x) // 12)
    plt.plot(x,mixed_means_s,linestyle="dotted",linewidth=3,color="black",label="Mixed training")
    plt.fill_between(x, lower, upper, alpha=0.05)

    #Finishing the graph
    plt.xlabel("Training Games Played")
    plt.ylabel("Win Rate")
    plt.title(f"Model trained using {model_class} learning curve (evaluated vs {evaluate_against})")
    plt.legend()
    plt.xlim(min(x), max(x))
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Model trained using {model_class} learning curve (evaluated vs {evaluate_against}).png")
    plt.close()
