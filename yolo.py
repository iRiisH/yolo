from layers import *

class yolo:

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters

        self.buildnet()

        fromvgg16 = self.hyperparameters.fromvgg16

        #load weights from vgg16 with random initialisation
        #of the last few layers
        if(fromvgg16):

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
                input_size = cfg_file.readline().strip().split(',')
                self.input_init(layer_name, input_size)
                continue

            if(layer_type == "convolutional"):
                continue

            if(layer_type == "max_pool"):
                continue

            if(layer_type == "fully_connected"):
                continue

            if(layer_type == "dropout"):
                continue
