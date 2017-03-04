# This is the main function

from hyperparameter import define_flags
from yolo import yolo
from train import train

hyperparameters = define_flags().FLAGS

network = yolo(hyperparameters)

network.train()

# network.predict()
