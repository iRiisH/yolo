import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import os
from cython_utils.cy_yolo_findboxes import yolo_box_constructor

def process_box(self, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = self.classes[max_indx]
    if max_prob > threshold:
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

def findboxes(self, net_out):
    # meta, FLAGS = self.meta, self.FLAGS
    threshold = self.hyperparameters.threshold

    boxes = []
    boxes = yolo_box_constructor(self, net_out, threshold)

    return boxes

def postprocess(self, net_out, im, save = True):
    """
    Takes net output, draw predictions, save to disk
    """
    # meta, FLAGS = self.meta, self.FLAGS
    threshold = self.hyperparameters.threshold
    # colors = meta['colors']
    labels = self.classes

    boxes = findboxes(self,net_out)

    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im

    h, w, _ = imgcv.shape
    textBuff = "["
    for b in boxes:
        boxResults = process_box(self, b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        if self.hyperparameters.json:
            line =     ('{"label": "%s",'
                    '"confidence": %.2f,'
                    '"topleft": {"x": %d, "y": %d},'
                    '"bottomright": {"x": %d,"y": %d}}, \n') % \
                    (mess, confidence, left, top, right, bot)
            textBuff += line
            continue

        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            self.colors[max_indx], thick)
        cv2.putText(
            imgcv, mess, (left, top - 12),
            0, 1e-3 * h, self.colors[max_indx],
            thick // 3)


    if not save: return imgcv

    # Removing trailing comma+newline adding json list terminator.
    textBuff = textBuff[:-2] + "]"
    outfolder = os.path.join(self.hyperparameters.test, 'out')
    img_name = os.path.join(outfolder, im.split('/')[-1])
    if self.hyperparameters.json:
        textFile = os.path.splitext(img_name)[0] + ".json"
        with open(textFile, 'w') as f:
            f.write(textBuff)
        return

    cv2.imwrite(img_name, imgcv)
