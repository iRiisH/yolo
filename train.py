from save import _save_ckpt

def train(self):
    loss_ph = self.placeholders
    profile = list ()
    loss_op = self.loss

    self.ann = self.parse_annotation(self.hyperparameters.ann_directory, self.classes)

    input_packs = self.read_data()

    for i,(x_batch, feed_batch) in enumerate(input_packs):
        #if not i: self.say(train_stats.format(
        #    self.FLAGS.lr, self.FLAGS.batch,
        #    self.FLAGS.epoch, self.FLAGS.save
        #))

        feed_dict = {
            loss_ph[key]: feed_batch[key]
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch_size)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)
