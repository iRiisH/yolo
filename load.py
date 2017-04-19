# load weights function

import numpy as np
import tensorflow as tf
import os

def loadFromVgg16(self):

    parameters_data = np.load(self.hyperparameters.vgg16_weights_file)
    keys = sorted(parameters_data.keys())
    vgg16_size = self.vgg16_size
    for i, k in enumerate(keys):
        if(i < vgg16_size):
            self.sess.run(self.parameters[i].assign(parameters_data[k]))
            print i, k, np.shape(parameters_data[k])
        else:
            break
    # self.sess.close()

def loadFromPreviousVersion(self):
    if self.hyperparameters.load < 0: # load lastest ckpt
        with open(self.hyperparameters.checkpoint_directory + 'checkpoint', 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            load_point = load_point.split('-')[-1]
            self.hyperparameters.load = int(load_point)

    load_point = os.path.join(self.hyperparameters.checkpoint_directory, 'yolo-psc')
    load_point = '{}-{}'.format(load_point, self.hyperparameters.load)
    # self.say('Loading from {}'.format(load_point))
    # try:
    self.saver.restore(self.sess, load_point)
    # except: load_old_graph(self, load_point)
