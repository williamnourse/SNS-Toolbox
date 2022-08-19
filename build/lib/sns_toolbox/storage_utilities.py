"""
Utility functions for saving and loading compiled SNS networks.
"""
from sns_toolbox.backends import Backend

import pickle

def save(model: Backend, filename: str) -> None:
    pickle.dump(model, open(filename, 'wb'))

def load(filename) -> Backend:
    model = pickle.load(open(filename, 'rb'))
    return model
