import os
import math
import numpy as np
import time

def is_inp(name):
    return name[-4:] in ['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def predict(self):
    inp_path = self.hyperparameters.test
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.hyperparameters.batch_size, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.layers['input'] : np.concatenate(inp_feed, 0)}
        # self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.layers['output'], feed_dict)
        stop = time.time(); last = stop - start
        # self.say('Total time = {}s / {} inps = {} ips'.format(
        #     last, len(inp_feed), len(inp_feed) / last))


        # Post processing
        # self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            # print str(i)
            # for j in prediction:
            #     print j

            self.postprocess(prediction,
                os.path.join(inp_path, this_batch[i]))
        stop = time.time(); last = stop - start

        # Timing
        # self.say('Total time = {}s / {} inps = {} ips'.format(
        #     last, len(inp_feed), len(inp_feed) / last))
