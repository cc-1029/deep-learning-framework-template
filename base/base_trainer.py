import abc


class BaseTrainer:
    def __init__(self, **kwargs):
        self.set_dict_attrs(kwargs)

    def set_dict_attrs(self, res_dict):
        for k, v in res_dict.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def init_components(self, parser):
        res_dict = {}
        for k, v in parser.config.items():
            if k != 'config_args' and k != 'trainer':
                if type(v) is not list:
                    res_obj = parser.init_obj(k)
                else:
                    res_obj = parser.init_objs(k)
                res_dict[k] = res_obj
        for k, v in parser.config_args.__dict__.items():
            res_dict[k] = v
        self.set_dict_attrs(res_dict)

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError