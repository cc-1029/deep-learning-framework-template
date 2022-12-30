import base_pt


class PtCifar10Trainer(base_pt.PtTrainer):
    def _train_step(self, train_data):
        # 定义如何取数据
        inputs, labels = train_data
        # 定义forward结果如何取
        outputs = self.model(inputs)
        return self.loss(outputs, labels)
