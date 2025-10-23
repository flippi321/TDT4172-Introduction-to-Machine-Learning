# models.py
from keras.models import Sequential
from keras.layers import Dense, Input
from keras import utils as kutils

def set_seed(seed: int = 696969) -> None:
    """
    Set random seed. Preferably a funny one
    """
    kutils.set_random_seed(seed)

def build_binary_mlp(input_dim: int = 2) -> Sequential:
    """
    Constructs a MLP network
    """
    model = Sequential(name="binary_mlp")
    model.add(Input(shape=(input_dim,), name="input"))
    model.add(Dense(16, activation='relu', name='hidden1'))
    model.add(Dense(32, activation='relu', name='hidden2'))
    model.add(Dense(64, activation='relu', name='hidden3'))
    model.add(Dense(32, activation='relu', name='hidden4'))
    model.add(Dense(16, activation='relu', name='hidden5'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    return model
