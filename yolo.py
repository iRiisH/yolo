from layers import *

class yolo:

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters

        self.buildnet()

        load_from_vgg16 = self.hyperparameters.load_from_vgg16

        #load weights from vgg16 with random initialisation
        #of the last few layers
        if(load_from_vgg16):

            self.loadFromVgg16()

        #load from the last trained version
        else:

            self.loadFromPreviousVersion()

    def buildnet(self):

        cfg_address = self.hyperparameters.cfg_address

        cfg_file = open(cfg_address, 'r')

        #to find the beginning of the cfg
        while(cfg_file.readline() != "#"):
            pass

        layer_type = ""
        layer_name = ""

        while(true):

            #to get rid of the empty line
            cfg_file.readline()

            # to note down the length of vgg16 network parameters
            if(layer_name == self.hyperparameters.last_vgg16_layer):
                self.vgg16_size = len(self.parameters)

            layer_type = cfg_file.readline()
            if(layer_type == "EOF"):
                break

            layer_name = cfg.file.readline()

            if(layer_type == "input"):
                # to convert string to int : map(int, array_of_str)
                input_size = map(int, cfg_file.readline().strip().split(','))
                self.input_init(layer_name, input_size)
                continue

            if(layer_type == "convolutional"):
                output_size = int(cfg_file.readline().strip())
                self.conv_layer(layer_name, output_size)
                continue

            if(layer_type == "max_pool"):
                self.max_pool_layer(layer_name)
                continue

            if(layer_type == "fully_connected"):
                output_size = int(cfg_file.readline().strip())
                self.fully_connected_layer(layer_name,output_size)
                continue

            if(layer_type == "dropout"):
                keep_prob = float(cfg_file.readline().strip())
                # it depends on whether we are training or predicting
                if(self.hyperparameters.action == 'train'):
                    self.dropout_layer(name, keep_prob)
                continue
