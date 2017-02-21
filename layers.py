import tensorflow as tf

def input_init(self, name, input_size):
    input_holder = tf.placeholder(
        tf.float32, [None, input_size[0], input_size[1], 3])
    self.layers[name] = input_holder
    self.last_output = input_holder

def conv_layer(self, name, weight_size):

    with tf.name_scope(name) as scope:

        kernel = tf.Variable(tf.truncated_normal([3,3,weight_size[0],weight_size[1]],dtype=float32,
                            stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(self.last_output, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[weight_size[1]], dtype=tf.float32),
                            trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)

        self.layers[name] = tf.nn.relu(out,name=scope)
        self.parameters += [kernel, biases]
        self.last_output = out

def max_pool(self, name):

    pool = tf.nn.max_pool(
        self.last_output,
        ksize = [1, 2, 2, 1],
        strides = [1, 2, 2, 1],
        padding = 'SAME',
        name = name))
    self.layers[name] = pool
    self.last_output = pool
