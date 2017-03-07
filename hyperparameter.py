from tensorflow import flags

def define_flags():

    # control parameters
    flags.DEFINE_string("load_from_vgg16", True, "decide whether load from vgg16 model or previously trained model")
    flags.DEFINE_string("action", "train", "to train, to predict, or ?")

    # output layer's name
    flags.DEFINE_string("last_vgg16_layer", "pool5", "name of last vgg16 layer")

    # network input output form constants
    flags.DEFINE_string("S", 7, "decide how we divide the pictures")
    flags.DEFINE_string("B", 2, "number of predictions per square")
    flags.DEFINE_string("batch_size", 64, "the size of batches (the number of pictures taken in each training circle)")

    # address
    flags.DEFINE_string("cfg_address", "cfg.txt", "address of the file for the cfg")
    flags.DEFINE_string("vgg16_weights_file", "vgg16/vgg16_weights.npz", "path to vgg16 weights file")
    flags.DEFINE_string("classes_file", "classes_file.txt", "the path to the file that contains classes' names")
    flags.DEFINE_string("ann_directory", "Annotations", "the path to the file that contains classes' names")

    # loss constants
    flags.DEFINE_string ("object_scale", 5, "" )
    flags.DEFINE_string ("noobject_scale", 1, "")
    flags.DEFINE_string ("class_scale", 1, "")
    flags.DEFINE_string("coord_scale", 1, "")

    #
    # flags.DEFINE_string ("absolute", 1, "")
    # flags.DEFINE_string ("thresh", 0.3, "")
    # flags.DEFINE_string ("random", 0, "")
    # flags.DEFINE_string ("im_side", 224, "")
    return flags
