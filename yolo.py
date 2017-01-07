import tensorflow as tf
import numpy as np
from vgg16.network import *
from scipy.misc import imread, imresize

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    network = vgg16(imgs, 'vgg16/vgg16_weights.npz', sess)

	# TODO modify network in order to implement YOLO neural network

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
