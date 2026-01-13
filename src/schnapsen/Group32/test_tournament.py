"""
Make a simple for loop test that Matches a specific version of a trained ML robot vs All other bots.

Gather statistics on wins/losses/draws and print after each game played to check progress as the tournament proceeds.

This is mainly to check that the trained bot is improving over time and not regressing.

Note: This is a simple test and does not require complex tournament structures or reporting.

"""

from schnapsen.game import Bot, Move, PlayerPerspective
from schnapsen.game import SchnapsenTrickScorer
from schnapsen.deck import Card, Suit, Rank
import pickle

# Placeholder for tournament test code


"""
%%time
for _ in range(x):
    eng.play_game(ml_data, rdeep, Random())
"""