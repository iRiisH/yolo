from tensorflow import flags

def define_flags():

    flags.DEFINE_string("fromvgg16", True, "decide whether load from vgg16 model")

    return flags
