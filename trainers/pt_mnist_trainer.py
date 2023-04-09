from backend import pt_backend


class PtMnistTrainer(pt_backend.PtTrainer):
    def _train_step(self, train_data):
        # 定义如何取数据
        inputs, labels = train_data[0].to(self.device), train_data[1].to(self.device)
        # 定义forward结果如何取
        outputs = self.model(inputs)
        print('lables.shape: ', labels.shape, '; outputs.shape: ', outputs.shape)
        return self.loss(outputs, labels)
