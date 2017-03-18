def train(self):

    self.ann = self.parse_annotation(self.hyperparameters.ann_directory, self.classes)

    input_packs = self.read_data()

    for i,current_pack in enumerate(input_packs):
        pass
