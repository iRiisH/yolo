

class yolo:

    def __init__(self,hyperparameters):

        self.hyperparameters = hyperparameters

        self.buildnet()

        fromvgg16 = hyperparameters.fromvgg16

        #load weights from vgg16 with random initialisation
        #of the last few layers
        if(fromvgg16):

            self.loadFromVgg16()

        #load from the last trained version
        else:

            self.loadFromPreviousVersion()
