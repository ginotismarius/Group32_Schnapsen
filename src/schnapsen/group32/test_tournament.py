"""
Make a simple for loop test that Matches a specific version of a trained ML robot vs All other bots.

Gather statistics on wins/losses/draws and print after each game played to check progress as the tournament proceeds.

This is mainly to check that the trained bot is improving over time and not regressing.

Note: This is a simple test and does not require complex tournament structures or reporting.

"""

import sys
import random
import itertools
from os.path import dirname as dname
from unittest import result

if __name__ == "__main__":
    # When run as a script add the toplevel src dir.
    srcdir = dname(dname(dname(__file__)))
    sys.path.append(srcdir)  # Add src/ to path

from schnapsen.game import Bot, Move, PlayerPerspective
from schnapsen.game import SchnapsenGamePlayEngine
from schnapsen.deck import Card, Suit, Rank

from schnapsen.bots.rand import RandBot
from schnapsen.bots.rdeep import RdeepBot
from schnapsen.bots.bully_bot import BullyBot


# Placeholder for tournament test code


"""
%%time
for _ in range(x):
    eng.play_game(ml_data, rdeep, Random())
"""

def bots_example1(rng: random.Random):
    return [
        RandBot(rng, 'RandomBot'),
        RdeepBot(10, 5, rng, 'RdeepBot'),
        BullyBot(rng, 'BullyBot'),
    ]

def bots_rdeepsonly1(rng: random.Random):
    return [
        RdeepBot(10, 3, rng, 'RdeepBot_shallow'),
        RdeepBot(10, 6, rng, 'RdeepBot_deep'),
    ]

def bots_rdeepsonly2(rng: random.Random):
    return [
        RdeepBot(10, 4, rng, 'RdeepBot_course'),
        RdeepBot(30, 4, rng, 'RdeepBot_fine'),
    ]


def tournament_all(bots: list[Bot], rng: random.Random, num_rounds: int):
    '''Every bot plays every other bot a given number of times 2'''
    botindices = range(len(bots))
    indexpairs = itertools.combinations(botindices, 2) # order is deterministic
    for i, j in indexpairs:
        # Typically bots are stateless so we don't need to copy them below...
        bot1 = bots[i]
        bot2 = bots[j]
        score1 = 0
        score2 = 0
        for _ in range(num_rounds):
            engine = SchnapsenGamePlayEngine()
            winner, points, score = engine.play_game(bot1, bot2, rng)
            if winner == bot1:
                score1 += 1
            else:
                score2 += 1

            # Run another game where bot2 is leader from start
            winner, points, score = engine.play_game(bot2, bot1, rng)
            if winner == bot2:
                score2 += 1
            else:
                score1 += 1

        print(f"Results between {bot1} and {bot2} over {num_rounds} rounds:")
        print(f"  {bot1} score: {score1}")
        print(f"  {bot2} score: {score2}")




# Add different tournament configurations here
TOURNAMENTS = {
    # name :        [seed, list of bots, tournament func,    args for tournament func]
    # --------       ----  ------------  ------------------  ------------------------
    'example1':     [1000, bots_example1, tournament_all, [10]],  # 1 rounds of every bot vs every other bot
    'example2':     [1001, bots_example1, tournament_all, [10]],  # Same as above but another seed...
    'example3':     [1002, bots_example1, tournament_all, [10]],  # Same as above but another seed...
    'rdeeponly1':   [1000, bots_rdeepsonly1, tournament_all, [50]],  # Rdeep bots only
    'rdeeponly2':   [1001, bots_rdeepsonly2, tournament_all, [50]],  # Rdeep bots only
}


def main(args):
    '''Parse args where first arg must be the tournament label to run'''
    if len(args) < 1:
        print("Please provide the tournament label to run.")
        return
    
    tourney_label = args[0]
    if tourney_label not in TOURNAMENTS:
        print(f"Tournament label '{tourney_label}' not recognized.")
        return
    
    seed, bot_func, tourney_func, tourney_args = TOURNAMENTS[tourney_label]
    rng = random.Random(seed)
    bots = bot_func(rng)
    tourney_func(bots, rng, *tourney_args)


if __name__ == "__main__":
    main(sys.argv[1:])