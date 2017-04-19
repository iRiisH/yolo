# This is the main function

from hyperparameter import define_flags
from yolo import yolo
from train import train

hyperparameters = define_flags().FLAGS

network = yolo(hyperparameters)

action = hyperparameters.action

if (action == 'train'):
    network.train()

elif (action == 'predict'):
    network.predict()
