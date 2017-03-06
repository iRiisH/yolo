import os

def _save_ckpt(self, step, loss_profile):
    filename = '{}-{}{}'
    model = 'yolonet'

    profile = filename.format(model, step, '.profile')
    profile = os.path.join(os.getcwd (), profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = filename.format(model, step, '')
    ckpt = os.path.join(os.getcwd (), ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)

def _load_ckpt(filename): # not to be opened in an existing session
	with tf.Session () as sess :
		saver = tf.train.Saver()
		saver.restore (sess, filename)
	