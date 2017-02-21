import tensorflow as tf
import numpy as np
from vgg16.network import *
from scipy.misc import imread, imresize
from vgg16.imagenet_classes import *

class yolonet:

    def __init__(data,vgg16_weights,sess):
        self.data = data
<<<<<<< HEAD
        self.vgg16(data,vvg16_weights,sess)
        self.addLayers()

    def addLayers(self):
        with tf.name_scope('conv_added1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
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
=======
        self.vgg16(data,vvg16_weights) # creates the vgg16 network, minus the FC layers, and load the weights
        self.addNewLayers()
>>>>>>> ca2b4e79ee434b103e69fe368a7c5f86305e0f37

    def vgg16(self, weights, sess):
        self.convlayers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
	# VGG16 convolutional layers
        self.parameters = []

        # zero-mean input
         with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

<<<<<<< HEAD
=======
    def fc_layers(self):
		# vgg16 final FC layers, not used in yolonet
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

>>>>>>> ca2b4e79ee434b103e69fe368a7c5f86305e0f37
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        length_parameters = len(self.parameters)
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            if(i < length_parameters):
                sess.run(self.parameters[i].assign(weights[k]))
	
	def addNewLayers(self):
        with tf.name_scope('conv_added1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,512,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
                                trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv_added = tf.nn.relu(out,name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv_added4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,1024,1024],dtype=float32,
                                stddev=1e-1),name='weights')
            conv = tf.nn.conv2d(self.conv_added3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32),
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
			
	def loss(self, net_out):
		# meta
		m = self.meta
		sprob = float(m['class_scale'])
		sconf = float(m['object_scale'])
		snoob = float(m['noobject_scale'])
		scoor = float(m['coord_scale'])
		S, B, C = m['side'], m['num'], m['classes']
		SS = S * S # number of grid cells

		size1 = [None, SS, C]
		size2 = [None, SS, B]

		# return the below placeholders
		_probs = tf.placeholder(tf.float32, size1)
		_confs = tf.placeholder(tf.float32, size2)
		_coord = tf.placeholder(tf.float32, size2 + [4])
		# weights term for L2 loss
		_proid = tf.placeholder(tf.float32, size1)
		# material calculating IOU
		_areas = tf.placeholder(tf.float32, size2)
		_upleft = tf.placeholder(tf.float32, size2 + [2])
		_botright = tf.placeholder(tf.float32, size2 + [2])

		self.placeholders = {
			'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
			'areas':_areas, 'upleft':_upleft, 'botright':_botright
		}

		# Extract the coordinate prediction from net.out
		coords = net_out[:, SS * (C + B):]
		coords = tf.reshape(coords, [-1, SS, B, 4])
		wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell
		area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2 
		centers = coords[:,:,:,0:2] # [batch, SS, B, 2]
		floor = centers - (wh * .5) # [batch, SS, B, 2]
		ceil  = centers + (wh * .5) # [batch, SS, B, 2]

		# calculate the intersection areas
		intersect_upleft   = tf.maximum(floor, _upleft) 
		intersect_botright = tf.minimum(ceil , _botright)
		intersect_wh = intersect_botright - intersect_upleft
		intersect_wh = tf.maximum(intersect_wh, 0.0)
		intersect = tf.mul(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])
    
		# calculate the best IOU, set 0.0 confidence for worse boxes
		iou = tf.truediv(intersect, _areas + area_pred - intersect)
		best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
		best_box = tf.to_float(best_box)
		confs = tf.mul(best_box, _confs)

		# take care of the weight terms
		conid = snoob * (1. - confs) + sconf * confs
		weight_coo = tf.concat(3, 4 * [tf.expand_dims(confs, -1)])
		cooid = scoor * weight_coo
		proid = sprob * _proid

		# flatten 'em all
		probs = slim.flatten(_probs)
		proid = slim.flatten(proid)
		confs = slim.flatten(confs)
		conid = slim.flatten(conid)
		coord = slim.flatten(_coord)
		cooid = slim.flatten(cooid)

		self.fetch += [probs, confs, conid, cooid, proid]
		true = tf.concat(1, [probs, confs, coord])
		wght = tf.concat(1, [proid, conid, cooid])

		print('Building {} loss'.format(m['model']))
		loss = tf.pow(net_out - true, 2)
		loss = tf.mul(loss, wght)
		loss = tf.reduce_sum(loss, 1)
		self.loss = .5 * tf.reduce_mean(loss)
				
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
    network = yolonet(imgs, 'vgg16/vgg16_weights.npz', sess)
	# TODO modify network in order to implement YOLO neural network

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(network.probs, feed_dict={network.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
