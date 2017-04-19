import tensorflow as tf
import layers
import load
import training_method
import train
import ann_parse
import loss
import data_reader
import image_preprocess
import predict
import postprocess

from helper import slicing

class yolo:

    # to pass the functions in layers.py as its methods
    input_init = layers.input_init
    conv_layer = layers.conv_layer
    max_pool_layer = layers.max_pool_layer
    fully_connected_layer = layers.fully_connected_layer
    dropout_layer = layers.dropout_layer
    output_layer = layers.output_layer

    # to pass the functions in load.py as its methods
    loadFromVgg16 = load.loadFromVgg16
    loadFromPreviousVersion = load.loadFromPreviousVersion

    # to pass the function in train.py as its methods
    train = train.train

    # to pass the loss function in loss.py
    defineloss = loss.defineloss

    # pass traing methods getter
    get_training_methods_set = training_method.get_training_methods_set

    # function in ann_parse
    parse_annotation = ann_parse.parse_annotation

    # function in data_reader
    read_data = data_reader.read_data
    get_image_data = data_reader.get_image_data

    # pass preprocess
    preprocess = image_preprocess.preprocess
    resize_input = image_preprocess.resize_input

    # predict
    predict = predict.predict

    # postprocess
    postprocess = postprocess.postprocess

    # a dict of layers in the network
    layers = dict()

    # a list of weights and bias
    parameters = list()

    # a dict of output variables
    outputs = dict()

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters

        self.setupsess()

        self.getclasses()

        self.buildnet()

        self.definetrain()

        self.sess.run(tf.global_variables_initializer())

        self.setupsaver()

        load_from_vgg16 = self.hyperparameters.load_from_vgg16

        #load weights from vgg16 with random initialisation
        #of the last few layers
        if(load_from_vgg16):

            self.loadFromVgg16()

        # #load from the last trained version
        else:

            self.loadFromPreviousVersion()


    def getclasses(self):

        path = self.hyperparameters.classes_file

        f = open(path,'r')

        self.classes = list()

        line = ''

        while(True):

            line = f.readline().strip()

            if(not line or line == ''):
                break

            self.classes.append(line)

    def buildnet(self):

        input_init = layers.input_init
        conv_layer = layers.conv_layer
        max_pool_layer = layers.max_pool_layer
        fully_connected_layer = layers.fully_connected_layer
        dropout_layer = layers.dropout_layer

        cfg_address = self.hyperparameters.cfg_address

        cfg_file = open(cfg_address, 'r')

        #to find the beginning of the cfg
        while(True):
            if(striped_line(cfg_file) == "#"):
                break

        layer_type = ""
        layer_name = ""

        while(True):

            #to get rid of the empty line
            striped_line(cfg_file)

            # to note down the length of vgg16 network parameters
            if(layer_name == self.hyperparameters.last_vgg16_layer):
                self.vgg16_size = len(self.parameters)

            layer_type = striped_line(cfg_file)
            if(layer_type == "EOF"):
                break

            layer_name = striped_line(cfg_file)

            print layer_name + ' is being created...'

            if(layer_type == "input"):
                # to convert string to int : map(int, array_of_str)
                self.input_size = map(int, striped_line(cfg_file).split(','))
                self.input_init(layer_name, self.input_size)
                continue

            if(layer_type == "convolutional"):
                output_size = int(striped_line(cfg_file))
                self.conv_layer(layer_name, output_size)
                continue

            if(layer_type == "max_pool"):
                self.max_pool_layer(layer_name)
                continue

            if(layer_type == "fully_connected"):
                if layer_name == "fully_connected_out" :
                    S = self.hyperparameters.S
                    B = self.hyperparameters.B
                    nClasses = len(self.classes)
                    output_size =  S * S * (nClasses + B * int(striped_line(cfg_file)))
                else :
                    output_size = int(striped_line(cfg_file))
                self.fully_connected_layer(layer_name,output_size)
                continue

            if(layer_type == "dropout"):
                keep_prob = float(striped_line(cfg_file))
                # it depends on whether we are training or predicting
                if(self.hyperparameters.action == 'train'):
                    self.dropout_layer(layer_name, keep_prob)
                continue

            if(layer_type == "output"):
                self.output_layer(layer_name)
                continue

    def definetrain(self):

        trainer = self.hyperparameters.training_method
        learning_rate = self.hyperparameters.learning_rate
        self.get_training_methods_set()

        self.defineloss(self.output)
        optimizer = self.training_methods[trainer](learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(gradients)

    def setupsess(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })
        utility = min(self.hyperparameters.gpu, 1.)
        if utility > 0.0:
            # self.say('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction = utility)
            cfg['allow_soft_placement'] = True
        else:
            # self.say('Running entirely on CPU')
            cfg['device_count'] = {'GPU': 0}

        # if self.FLAGS.train: self.build_train_op()

        self.summary_op = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')

        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        # self.sess.run(tf.global_variables_initializer())

        # if not self.ntrain: return
        # self.saver = tf.train.Saver(tf.global_variables(),
        #     max_to_keep = self.FLAGS.keep)
        # if self.FLAGS.load != 0: self.load_from_ckpt()

        # self.writer.add_graph(self.sess.graph)

    def setupsaver(self):
        self.saver = tf.train.Saver(tf.global_variables(),
            max_to_keep = self.hyperparameters.keep)
        # if self.FLAGS.load != 0: self.load_from_ckpt()

def striped_line(file):
    return file.readline().strip()
