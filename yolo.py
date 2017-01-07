import tensorflow as tf
import numpy as np
from vgg16.network import *
from scipy.misc import imread, imresize
from vgg16.imagenet_classes import *

class yolonet:

    def __init__(data,vgg16_weights,sess):
        self.data = data
        self.vgg16(data,vvg16_weights,sess)
        self.deleteLayers()
        self.addLayers()

    def deleteLayers():
        #TODO
        return

    def addLayers(self):
        with tf.name_scope('conv_added1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,512],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.conv_added4.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape,4096],dtype=tf.float32,
                                stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096],dtype=tf.float32),
                                trainable=True, name='biases')
            conv_added4_flat = tf.reshape(self.conv_added4, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(conv_added4_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        keep_prob = tf.placeholder(float32)
        self.dropout=tf.nn.dropout(self.fc1,keep_prob)

        with tf.name_scope('fc2') as scope:
            shape = int(np.prod(self.dropout.get_shape()[1:]))
            fc2w = tf.Variable(tf.truncated_normal([shape, 4096],dtype=tf.float32,
                                stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable=True, name='biases')
            dropout_flat = tf.reshape(self.dropout, [-1, shape])
            fc2l = tf.nn.bias_add(tf.matmul(dropout_flat, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

if __name__ == '__main__':

    #new code
    # sess = tf.Session()
    # #TODO how to change the size of inputs
    # data = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # network = yolonet(data,'vgg16/vgg16_weights.npz',sess)
    #
    # data_input = imread('laska.png', mode='RGB')
    # data_input = imresize(data, (224, 224))
    #
    # #TODO is it the same for several images?
    # prob = sess.run(network.probs, feed_dict={network.data: [data_input]})[0]
    # preds = (np.argsort(prob)[::-1])[0:5]
    # for p in preds:
    #     print class_names[p], prob[p]

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    network = vgg16(imgs, 'vgg16/vgg16_weights.npz', sess)
	# TODO modify network in order to implement YOLO neural network

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(network.probs, feed_dict={network.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
