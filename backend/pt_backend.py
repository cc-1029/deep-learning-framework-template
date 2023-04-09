import random

import numpy as np
import torch

from base import base_trainer

class PtTrainer(base_trainer.BaseTrainer):
    """Base trainers class for PyTorch

    """
    def init_components(self, parser):
        res_dict = {}
        for k, v in parser.config.items():
            if k != 'config_args' and k != 'trainer' and k != 'optimizer':
                if type(v) is not list:
                    res_obj = parser.init_obj(k)
                else:
                    res_obj = parser.init_objs(k)
                res_dict[k] = res_obj
        res_dict['optimizer'] = parser.init_obj('optimizer', res_dict['model'].parameters())
        for k, v in parser.config_args.__dict__.items():
            res_dict[k] = v
        self.set_dict_attrs(res_dict)

    def train(self):
        self.set_devices()
        self.model.to(self.device)
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for step, train_data in enumerate(self.train_dataloader, 1):
                self.optimizer.zero_grad()
                loss = self._train_step(train_data)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if step % 2000 == 0:  # print every 2000 mini-batches
                    # self.logger.info(f'[{epoch}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        # torch.save(self.model, self.saved_model_path)

    def set_devices(self):
        gpu_list = try_all_gpus()
        if len(gpu_list) > 0:
            self.device = gpu()
        else:
            self.device = cpu()

    def _train_step(self, train_data):
        raise NotImplementedError

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cpu():
    return torch.device('cpu')


def gpu(i=0):
    return torch.device(f'cuda:{i}')


def num_gpus():
    return torch.cuda.device_count()


def try_gpu(i=0):
    if num_gpus() > i:
        return gpu(i)
    return cpu()


def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]