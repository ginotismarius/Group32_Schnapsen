"""
%%time
train_ML_model(Path('ML_replay_memories') / 'bully_replay_memory', Path('ML_models') / 'bully_model', "LR")

"""

from schnapsen.game import Bot, Move, PlayerPerspective
from schnapsen.game import SchnapsenTrickScorer
from schnapsen.deck import Card, Suit, Rank
import pickle


# From Group32 push test

# Change made in Marius branch
# Test 2