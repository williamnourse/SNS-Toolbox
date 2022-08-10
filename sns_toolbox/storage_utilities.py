"""
Utility functions for saving and loading compiled SNS networks.
"""
from sns_toolbox.backends import __Backend_New__

import pickle

def save(model: __Backend_New__, filename: str) -> None:
    pickle.dump(model, open(filename, 'wb'))

def load(filename) -> __Backend_New__:
    model = pickle.load(open(filename, 'rb'))
    return model
