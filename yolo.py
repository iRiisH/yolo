import tensorflow as tf
import layers
import load
import train
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

    # a dict of layers in the network
    layers = dict()

    # a list of weights and bias
    parameters = list()

    # a dict of output variables
    outputs = dict()

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters

        self.getclasses()

        self.buildnet()

        load_from_vgg16 = self.hyperparameters.load_from_vgg16

        self.defineloss()
        #load weights from vgg16 with random initialisation
        #of the last few layers
        if(load_from_vgg16):

            self.loadFromVgg16()

        # #load from the last trained version
        # else:
        #
        #     self.loadFromPreviousVersion()

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
                print self.last_output.get_shape()
                break

            layer_name = striped_line(cfg_file)

            print layer_name + ' is being created...'

            if(layer_type == "input"):
                # to convert string to int : map(int, array_of_str)
                input_size = map(int, striped_line(cfg_file).split(','))
                self.input_init(layer_name, input_size)
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

    def defineloss(self,):

        pass
        # gt = dict()
        #
        # self.ground_truth = gt
        #
        # lobj = self.hyperparameters.object_scale
        # lnoobj = self.hyperparameters.noobject_scale
        # lcoord = self.hyperparameters.coord_scale
        # lclass = self.hyperparameters.class_scale
        #
        # S = self.hyperparameters.S
        # batch_size = self.hyperparameters.batch_size
        # num_classes = len(self.classes)
        #
        # gt['prob'] = tf.placeholder(tf.float32,shape=(batch_size,S,S,num_classes))
        # gt['hasObject'] = tf.placeholder()
        # gt['hasClass'] = tf.placeholder()
        # gt['coord'] = tf.placeholder(tf.float32,shape=(batch_size,S,S,8))
        # gt[]

def striped_line(file):
    return file.readline().strip()
