import abc
import inspect


class BaseModule:
    """Base class for all modules

    """

    def save_class_attributes(self, ignore=None):
        """Main save function arguments into class attributes.

        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        if ignore is None:
            ignore = []
        self.hparams = {
            k:v for k,v in local_vars.items() if k not in set(ignore + ['self']) and not k.startswith('_')
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


class BaseTrainer(BaseModule):
    """Base trainer class for all frameworks

    """
    def __init__(self, num_epochs, model, loss, optimizer, train_dataloader, eval_dataloader=None, metrics=None):
        self.save_class_attributes()

    @abc.abstractmethod
    def train(self):
        """Main interface: custom train loops for different frameworks

        """
        raise NotImplementedError