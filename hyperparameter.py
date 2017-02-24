from tensorflow import flags

def define_flags():

    flags.DEFINE_string("load_from_vgg16", True, "decide whether load from vgg16 model or previously trained model")

    flags.DEFINE_string("cfg_address", "cfg.txt", "address of the file for the cfg")

    flags.DEFINE_string("last_vgg16_layer", "pool5", "name of last vgg16 layer")

    flags.DEFINE_string("action", "train", "to train, to predict, or ?")

    flags.DEFINE_string("number_of_cases", 7, "decide how we divide the pictures")

    flags.DEFINE_string("vgg16_weights_file", "vgg16/vgg16_weights.npz", "path to vgg16 weights file")
    
    return flags
