import numpy as np
import os
import sys

from copy import deepcopy
from numpy.random import permutation as perm

def read_data(self):

    batch = self.hyperparameters.batch_size
    ann = self.ann
    size = len(ann)
    epoch = self.hyperparameters.epoch

    print('Dataset of {} instance(s)'.format(size))
    # if batch > size: self.FLAGS.batch = batch = size
    batch_per_epoch = int(size / batch)

    for i in range(epoch):
        shuffle_idx = perm(np.arange(size))
        for b in range(batch_per_epoch):
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b*batch, b*batch+batch):
                train_instance = ann[shuffle_idx[j]]
                inp, new_feed = self.get_image_data(train_instance)

                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key,
                        np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([
                        old_feed, [new]
                    ])

            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch


def get_image_data(self, chunk):
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's
    input & loss layer correspond to this chunk
    """
    # meta = self.meta
    # S, B = meta['side'], meta['num']
    # C, labels = meta['classes'], meta['labels']
    S = self.hyperparameters.S
    B = self.hyperparameters.B
    C = len(self.classes)
    classes = self.classes

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)
    # !!!!
    path = os.path.join(self.hyperparameters.image_directory, jpg)
    img = 0
    # img = self.preprocess(path, allobj)

    # data to fill
    data = dict()
    data['prob'] = np.zeros([S,S,C])
    data['hasObject'] = np.zeros([S,S,1])
    data['coord'] = np.zeros([S,S,4])
    data['IOU'] = np.zeros([S,S,1])
    data['x'] = data['coord'][:,:,0]
    data['y'] = data['coord'][:,:,1]
    data['w'] = data['coord'][:,:,2]
    data['h'] = data['coord'][:,:,3]

    # obj = [class, xmin, ymin, xmax, ymax]
    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S

    for obj in allobj:

        centerx = .5*(obj[1]+obj[3])
        centery = .5*(obj[2]+obj[4])

        # cx, cy relative positions to case units
        cx = centerx / cellx
        cy = centery / celly

        # if cx >= S or cy >= S: return None, None

        obj[3] = float(obj[3]-obj[1]) / w
        obj[4] = float(obj[4]-obj[2]) / h
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])

        # relative position in case
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery

        obj += [int(np.floor(cy) * S + np.floor(cx))]

    # obj = [class, x relative, y relative, sqrt(w)relative, sqrt(h)relative, indexOfCase]

    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values
    probs = np.zeros([S*S,C])
    confs = np.zeros([S*S,B])
    coord = np.zeros([S*S,B,4])
    proid = np.zeros([S*S,C])
    prear = np.zeros([S*S,4])

    for obj in allobj:


        probs[obj[5], :] = [0.] * C
        probs[obj[5], classes.index(obj[0])] = 1.


        proid[obj[5], :] = [1] * C
        coord[obj[5], :, :] = [obj[1:5]] * B
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * S # xleft
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * S # yup
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * S # xright
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * S # ybot
        confs[obj[5], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    wh = botright - upleft;
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer
    loss_feed_val = {
        'probs': probs, 'confs': confs,
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft,
        'botright': botright
    }

    # probabilities
    print 'probs'+str(loss_feed_val['probs'].shape)
    # IOU*P(hasObject)
    print 'confs'+str(loss_feed_val['confs'].shape)
    
    print 'coord'+str(loss_feed_val['coord'].shape)
    print 'areas'+str(loss_feed_val['areas'].shape)
    print 'proid'+str(loss_feed_val['proid'].shape)
    print 'upleft'+str(loss_feed_val['upleft'].shape)
    print 'botright'+str(loss_feed_val['botright'].shape)

    return inp_feed_val, loss_feed_val
