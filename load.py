# load weights function

import numpy as np
import tensorflow as tf

def loadFromVgg16(self):

    sess = tf.Session()
    parameters_data = np.load(self.hyperparameters.vgg16_weights_file)
    keys = sorted(parameters_data.keys())
    vgg16_size = self.vgg16_size
    for i, k in enumerate(keys):
        if(i < vgg16_size):
            sess.run(self.parameters[i].assign(parameters_data[k]))
            print i, k, np.shape(parameters_data[k])
        else:
            break
    sess.close()

def loadFromPreviousVersion(self):
    pass
