"""
File for gathering results from experiments and evaluating the performance of the models
"""

from schnapsen.game import Bot, SchnapsenGamePlayEngine
from schnapsen.bots import RandBot
from random import Random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Notes: eval_games, repeats only set with local defaults change Args for plot functions so they are sent and can be changed easier from experiment.py done?
# Finish Doc strings + args were missing
# So much reading T_T help

def evaluate_learning_multiple_opponents(bots: list[Bot],opponents: list[Bot],eval_games: int,repeats: int) -> tuple[list[float], list[float]]:
    """
    Evaluate a sequence of models against multiple opponents and compute the mean performance across all opponents.

    Args:
        something here
    Returns:
        mean_over_opponents: list[float]
        std_over_opponents: list[float]
    """
    all_means = []
    all_stds = []

    for opponent in opponents:
        means, stds = evaluate_learning(bots, opponent, eval_games, repeats)
        all_means.append(means)
        all_stds.append(stds)

    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    mean_over_opponents = np.mean(all_means, axis=0)
    std_over_opponents = np.sqrt(np.mean(all_stds**2, axis=0))

    return mean_over_opponents, std_over_opponents

def evaluate_learning(bots: list[Bot], opponent: Bot, eval_games: int, repeats: int) -> tuple[list[float], list[float]]:
    """
    Evaluate a sequence of bots representing successive training iterations.

    Args:
        bots: list[Bot] - List of a specific ML model
        opponent: Bot - The opponent we will be testing against
        eval_games: int - How many games we will run to get win rate statistics
        repeats: int - Amount of times we will rerun the test with different seeds
        used to eliminate "Lucky" games and keep the win rate based on skill.

    Returns:
        means: list[float] - List of the arithmetic mean along the specified axis.
        stds: list[float] - List of the standard deviation along the specified axis.
    """
    means, stds = [],[]

    for bot in bots:
        mean, std = evaluate_bot(bot,opponent,eval_games,repeats)
        means.append(mean)
        stds.append(std)

    return means, stds

def evaluate_bot(bot: Bot, opponent: Bot, eval_games: int, repeats: int) -> tuple[float, float]:
    """
    Evaluate a bot by playing repeated games against a fixed opponent.

    Args:
        bot: Bot - ML model we want to test
        opponent: Bot - The opponent we will be testing against
        eval_games: int - How many games we will run to get win rate statistics (default =400)
        repeats: int - Amount of times we will rerun the test with different seeds (default =10)
        used to eliminate "Lucky" games and keep the win rate based on skill.
    
    Returns: 
        mean_win_rate: float - The arithmetic mean along the specified axis.
        std_win_rate: float - The standard deviation along the specified axis.
    """ 
    eng = SchnapsenGamePlayEngine()
    win_rates = []
    print(f"Evaluating: {bot} vs {opponent}")
    for seed in range(repeats):
        rng = Random(seed)
        wins = 0

        for _ in range(eval_games):
            winner, _, _ = eng.play_game(bot, opponent, rng)
            if winner is bot:
                wins += 1  
        
        win_rates.append(wins/eval_games)
    
    return float(np.mean(win_rates)), float(np.std(win_rates))

def evaluate_final_model_iteration_matrix(bots: list[Bot], eval_games: int, repeats: int) -> pd.DataFrame:
    """
    Evaluate final trained models against each other.

    Args:
        bots: list[Bot] - List of trained bots (final iteration models).
        eval_games: int - How many games we will run to get win rate statistics (default =400)
        repeats: int - Amount of times we will rerun the test with different seeds (default =10)

    Returns:
        pd.DataFrame: - Win-rate matrix where rows are evaluated bots and columns are opponents.
    """
    results = pd.DataFrame(index=bots, columns=bots, dtype=float)

    for bot in bots:
        for opponent in bots:
            if bot == opponent:
                results.loc[bot, opponent] = None
                continue
            mean_win, _ = evaluate_bot(bot,opponent,eval_games,repeats)
            results.loc[bot, opponent] = mean_win

    return results

def smooth_curve(y, window_size):
    """
    Smoothing the learning plot curves by taking the x:(window_size) neibouring points into consideration.
    Uses only available points near the boundaries.
    Preserves first and last points.
    """
    y = np.asarray(y, dtype=float)
    smoothed = np.empty_like(y)

    half = window_size // 2

    for i in range(len(y)):
        start = max(0, i - half)
        end = min(len(y), i + half + 1)
        smoothed[i] = np.mean(y[start:end])

    return smoothed

def confidence_intervals(mean,std,z=1.96):
    """
    Getting the standard deviation in plots to represents the variability or dispersion of data around a mean value
    Love finding out when reading mathlib plot documentation that there was an easier way to do this by using plt.errorbar WeeWooWeeWoo slowly losing it.

    68% CI → z = 1.0
    90% CI → z ≈ 1.645
    95% CI → z ≈ 1.96
    99% CI → z ≈ 2.576

    """
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
        smooth_window: int =2,
        eval_games:int = 100,
        repeats:int= 10
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
        smooth_window: int - Neighbouring points to compare to amount (default = 2)
        eval_games: int - How many games we will run to get win rate statistics (default =100)
        repeats: int - Amount of times we will rerun the test with different seeds (default =10)
    """
    plt.figure(figsize=(12, 8))

    games_per_iteration = total_games // iterations
    drop = smooth_window // 2

    x = [0] + [min((i + 1) * games_per_iteration, total_games) for i in range(len(mixed_models))]
    
    baseline_bot = RandBot(Random(0), "Baseline")
    baseline_mean, baseline_std = evaluate_bot(baseline_bot, evaluate_against, eval_games, repeats)

    # Single opponent learning curves
    for opponent, bots in single_models_by_opponent.items():

        means, stds = evaluate_learning(bots, evaluate_against, eval_games, repeats)

        means = [baseline_mean] + means
        stds = [baseline_std] + stds

        means_s = smooth_curve(means, smooth_window)
        stds_s = smooth_curve(stds, smooth_window)

        lower, upper = confidence_intervals(means_s, stds_s)

        plt.plot(x, means_s, linestyle="solid", linewidth=3, label=f"Trained vs {opponent}")
        plt.fill_between(x, lower, upper, alpha=0.03)

    # Mixed opponent learning curves
    mixed_means, mixed_stds = evaluate_learning(mixed_models, evaluate_against, eval_games, repeats)

    mixed_means = [baseline_mean] + mixed_means
    mixed_stds = [baseline_std] + mixed_stds

    mixed_means_s =smooth_curve(mixed_means, smooth_window)
    mixed_stds_s = smooth_curve(mixed_stds, smooth_window)

    lower, upper = confidence_intervals(mixed_means_s, mixed_stds_s)

    plt.plot(x,mixed_means_s,linestyle="dotted",linewidth=3,color="black",label="Mixed training")
    plt.fill_between(x, lower, upper, alpha=0.05)
    #--------------------------------------------
    plt.xlabel("Training Games Played")
    plt.ylabel("Win Rate")
    plt.title(f"Model trained using {model_class} learning curve (evaluated vs {evaluate_against})")
    plt.legend()
    plt.xlim(0,total_games)
    plt.ylim(0,1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Model trained using {model_class} learning curve (evaluated vs {evaluate_against}).png")
    plt.close()

def plot_generalization_curve(
        model_class: str,
        evaluation_opponents: list[Bot],
        single_models_by_opponent: dict[str, list[Bot]],
        mixed_models: list[Bot],
        total_games: int,
        iterations: int,
        smooth_window: int = 2,
        eval_games:int = 100,
        repeats: int = 10
        ):
    """
    Plot learning curves for single-opponent and mixed-opponent training evaluated against all opponents

    Args:
        model_class: str - What was used when training the ML model "LR"|"NN"
        evaluation_opponents: list[Bot] - List of all opponent bots to test on
        single_models_by_opponent: dict[str,list[Bot]] - A dictonary where key is what the bot was trained against.
        And values being a list of the Model iterations
        mixed_models: list[Bot] - A list of models trained against mixed oponents with their iterations
        total_games: int - Total games used in training
        iterations: Amount of iterations saved of the models
        smooth_window: int - Neighbouring points to compare to amount (default = 2)
        eval_games: int - How many games we will run to get win rate statistics (default =100)
        repeats: int - Amount of times we will rerun the test with different seeds (default =10)
    """
    plt.figure(figsize=(12, 8))

    games_per_iteration = total_games // iterations
    drop = smooth_window // 2

    x = [0] + [min((i + 1) * games_per_iteration, total_games) for i in range(len(mixed_models))]

    baseline_bot = RandBot(Random(0), "Baseline")
    baseline_means, baseline_stds = evaluate_learning_multiple_opponents([baseline_bot], evaluation_opponents, eval_games, repeats)
    baseline_mean = baseline_means[0]
    baseline_std = baseline_stds[0]

    # Single-opponent trained models
    for opponent, bots in single_models_by_opponent.items():

        means, stds = evaluate_learning_multiple_opponents(bots, evaluation_opponents, eval_games, repeats)

        means = np.concatenate(([baseline_mean], means))
        stds  = np.concatenate(([baseline_std], stds))

        means_s = smooth_curve(means, smooth_window)
        stds_s = smooth_curve(stds, smooth_window)
        
        lower, upper = confidence_intervals(means_s, stds_s)

        plt.plot(x, means_s, linestyle="solid", linewidth=3, label=f"Trained vs {opponent}")
        plt.fill_between(x, lower, upper, alpha=0.05)

    # Mixed training
    mixed_means, mixed_stds = evaluate_learning_multiple_opponents(mixed_models, evaluation_opponents, eval_games, repeats)

    mixed_means = np.concatenate(([baseline_mean], mixed_means))
    mixed_stds = np.concatenate(([baseline_std], mixed_stds))

    mixed_means_s = smooth_curve(mixed_means, smooth_window)
    mixed_stds_s = smooth_curve(mixed_stds, smooth_window)

    lower, upper = confidence_intervals(mixed_means_s, mixed_stds_s)

    plt.plot(x, mixed_means_s, linestyle="dotted",linewidth=3, color="black", label="Mixed training")
    plt.fill_between(x, lower, upper, alpha=0.05)

    #--------------------------------------------
    plt.xlabel("Training Games Played")
    plt.ylabel("Mean Win Rate")
    plt.title(f"{model_class} Mean Performance against All Opponents")
    plt.legend()
    plt.grid(True)
    plt.xlim(0,total_games)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(f"{model_class}_learning_curve_mean.png")
    plt.close()

def plot_model_heatmap(bots:list[Bot], title="Model Win-Rate Heatmap", eval_games:int=100, repeats:int = 10):
    """
    Plot heatmaps for evaluating model vs model performance
    """
    results_df = evaluate_final_model_iteration_matrix(bots)
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_df,annot=True,fmt=".2f",cmap="magma",vmin=0,vmax=1,linewidths=0.5,cbar_kws={"label": "Win Rate"})
    plt.title(title)
    plt.xlabel("Opponent")
    plt.ylabel("Testing Model")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()
