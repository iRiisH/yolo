import numpy as np
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2

def preprocess(self, im, allobj = None):

    sys.path.append('/usr/local/lib/python2.7/site-packages')

    """
    Takes an image, return it as a numpy tensor that is readily
    to be fed into tfnet. If there is an accompanied annotation (allobj),
    meaning this preprocessing is serving the train process, then this
    image will be transformed with random noise to augment training data,
    using scale, translation, flipping and recolor. The accompanied
    parsed annotation (allobj) will also be modified accordingly.
    """
    if type(im) is not np.ndarray:
        im = cv2.imread(im)

    if allobj is not None: # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip: continue
            obj_1_ =  obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        im = imcv2_recolor(im)

    im = self.resize_input(im)
    if allobj is None: return im
    return im#, np.array(im) # for unit testing

def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)

def resize_input(self, im):
    h, w, c = self.input_size
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

def imcv2_recolor(im, a = .1):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.

    # random amplify each channel
    im = im * (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform() * 2 - 1
    im = np.power(im/mx, 1. + up * .5)
    return np.array(im * 255., np.uint8)

def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0,0), fx = scale, fy = scale)
    im = im[offy : (offy + h), offx : (offx + w)]
    flip = np.random.binomial(1, .5)
    if flip: im = cv2.flip(im, 1)
    return im, [w, h, c], [scale, [offx, offy], flip]
