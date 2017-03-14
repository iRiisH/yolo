import ann_parse

def train(self):

    self.ann = self.parse_annotation(self.hyperparameters.ann_directory, self.classes)

    print self.ann[1]

    # input_package = data_reader()

    pass
