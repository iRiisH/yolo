import os
import pickle

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = 'yolo-psc'

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.hyperparameters.checkpoint_directory, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.hyperparameters.checkpoint_directory, ckpt)
    # self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)

def _load_ckpt(filename): # not to be opened in an existing session
    with tf.Session () as sess :
        saver = tf.train.Saver()
        saver.restore (sess, filename)
