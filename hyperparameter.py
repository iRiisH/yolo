from tensorflow import flags

def define_flags():

    flags.DEFINE_string("fromvgg16", True, "decide whether load from vgg16 model")

    flags.DEFINE_string("cfg_address", "cfg.txt", "address of the file for the cfg")

    flags.DEFINE_string("last_vgg16_layer", "", "name of last vgg16 layer")

    

    return flags
