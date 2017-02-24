import tensorflow as tf
import numpy as np

def input_init(self, name, input_size):
    input_holder = tf.placeholder(
        tf.float32, [None, input_size[0], input_size[1], 3])
    self.layers[name] = input_holder
    self.last_output = input_holder

def conv_layer(self, name, output_size):

    with tf.name_scope(name) as scope:

        input_size = int(self.last_output.get_shape()[-1])
        kernel = tf.Variable(tf.truncated_normal([3,3,input_size,output_size],dtype=tf.float32,
                            stddev=1e-1),name='weights')
        prod = tf.nn.conv2d(self.last_output, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[output_size], dtype=tf.float32),
                            trainable=True, name='biases')
        conv = tf.nn.relu(tf.nn.bias_add(prod, biases), name=scope)

        self.layers[name] = conv
        self.parameters += [kernel, biases]
        self.last_output = conv

def max_pool_layer(self, name):

    pool = tf.nn.max_pool(
        self.last_output,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME',
        name = name)

    self.layers[name] = pool
    self.last_output = pool

def fully_connected_layer(self,name,output_size):
    with tf.name_scope(name) as scope:
        input_size = int(np.prod(self.last_output.get_shape()[1:]))
        weights = tf.Variable(tf.truncated_normal([input_size,output_size],dtype=tf.float32,
                            stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(1.0, shape=[output_size],dtype=tf.float32),
                            trainable=True, name='biases')
        reshaped_last_output = tf.reshape(self.last_output, [-1, input_size])
        fully_connected = tf.nn.relu(tf.nn.bias_add(
            tf.matmul(reshaped_last_output, weights), biases),name=scope)

        self.layers[name] = fully_connected
        self.parameters += [weights, biases]
        self.last_output = fully_connected

def dropout_layer(self, name, keep_prob):
    dropout=tf.nn.dropout(self.last_output, keep_prob)
    self.layers[name] = dropout
    self.last_output = dropout

def output_layer(self, name):
    
