from save import _save_ckpt

def train(self):

    loss_mva = None
    loss_ph = self.placeholders
    profile = list ()
    loss_op = self.loss

    self.ann = self.parse_annotation(self.hyperparameters.ann_directory, self.classes)

    input_packs = self.read_data()

    for i,(x_batch, feed_batch) in enumerate(input_packs):

        feed_dict = {
            loss_ph[key]: feed_batch[key]
                for key in loss_ph }
        feed_dict[self.layers['input']] = x_batch
        # feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.hyperparameters.load + i + 1

        # form = 'step {} - loss {} - moving ave loss {}'
        # self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.hyperparameters.checkpoint // self.hyperparameters.batch_size)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)
