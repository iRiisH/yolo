from tensorflow import flags

def define_flags():

    # control parameters
    flags.DEFINE_boolean ("load_from_vgg16", True, "decide whether load from vgg16 model or previously trained model")
    flags.DEFINE_string ("action", "train", "to train, to predict, or ?")
    flags.DEFINE_string("load", "", "how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
    flags.DEFINE_integer("save", 2000, "save checkpoint every ? training examples")
    flags.DEFINE_float("learning_rate", 1e-5, "learning rate")

    # output layer's name
    flags.DEFINE_string ("last_vgg16_layer", "pool5", "name of last vgg16 layer")

    # network input output form constants
    flags.DEFINE_integer ("S", 7, "decide how we divide the pictures")
    flags.DEFINE_integer ("B", 2, "number of predictions per square")
    flags.DEFINE_integer ("batch_size", 1, "the size of batches (the number of pictures taken in each training circle)")

    # address
    flags.DEFINE_string ("cfg_address", "cfg.txt", "address of the file for the cfg")
    flags.DEFINE_string ("vgg16_weights_file", "vgg16/vgg16_weights.npz", "path to vgg16 weights file")
    flags.DEFINE_string ("classes_file", "classes_file.txt", "the path to the file that contains classes' names")
    # flags.DEFINE_string ("ann_directory", "Annotations", "the path to the file that contains classes' names")
    # flags.DEFINE_string ("ann_parsed_file", "ann_parsed_file", "the path to the file that contains ann_parsed")
    # flags.DEFINE_string ("image_directory", "test", "the path to the image directory")
    flags.DEFINE_string ("ann_directory", "test/ann", "the path to the file that contains classes' names")
    flags.DEFINE_string ("ann_parsed_file", "test/ann/ann_parsed_file", "the path to the file that contains ann_parsed")
    flags.DEFINE_string ("image_directory", "test", "the path to the image directory")

    # loss constants
    flags.DEFINE_float ("object_scale", 5.0, "" )
    flags.DEFINE_float ("noobject_scale", 1.0, "")
    flags.DEFINE_float ("class_scale", 1.0, "")
    flags.DEFINE_float ("coord_scale", 1.0, "")

    # parameters for training
    flags.DEFINE_string ("training_method", "adagrad", "indicate which training methods to use")
    flags.DEFINE_integer ("epoch", 1, "numbers of training turns")

    #
    # flags.DEFINE_string ("absolute", 1, "")
    # flags.DEFINE_string ("thresh", 0.3, "")
    # flags.DEFINE_string ("random", 0, "")
    # flags.DEFINE_string ("im_side", 224, "")
    return flags
