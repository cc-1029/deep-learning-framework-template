import abc
import argparse
import importlib
import json

class ConfigParser:
    def __init__(self, config):
        self._config = config
    
    @classmethod
    def from_cli_args(cls, cli_args):
        """
        Construct method from cli args
        Args:
            cli_args: Arguments from the command line
        """
        assert isinstance(cli_args, argparse.Namespace), 'cli args type error'
        assert 'config' in cli_args.__dict__, '"config" not in the key of cli args'
        config = cls.load_config(cls, cli_args.config)
        assert 'config_args' in config, '"config_args" not in the key of configs'
        # update the args from cli args
        for k, v in cli_args.__dict__.items():
            # modify the args based on the cli args
            if k != 'config':
                config['config_args'][k] = v
        return cls(config)
    
    @abc.abstractmethod
    def load_config(self, path):
        raise NotImplementedError
    
    @abc.abstractmethod
    def dump_config(self, path):
        raise NotImplementedError
    
    def __getitem__(self, key):
        return self.config[key]

    @property
    def config(self):
        return self._config
    
    @property
    def config_args(self):
        return argparse.Namespace(**self.config['config_args'])

    def init_trainer(self):
        trainer = self.init_obj('trainer')
        trainer.init_components(self)
        trainer.init_optimizer(self)
        return trainer
    
    def init_obj(self, name, *args, **kwargs):
        obj_name = self[name]
        return self._create_obj(obj_name, name, *args, **kwargs)

    def init_objs(self, name, *args, **kwargs):
        objs_name = self[name]
        return [self._create_obj(obj_name, name, *args, **kwargs) for obj_name in objs_name]

    def get_attr(self, name):
        obj_name = self[name]
        return self._create_attr(obj_name, name)

    def _create_obj(self, _dict, name, *args, **kwargs):
        try:
            _attr = self._create_attr(_dict, name)
            if 'kwargs' in _dict:
                dict_kwargs = dict(_dict['kwargs'])
                kwargs.update(dict_kwargs)
            if 'use_config_args' in _dict and _dict['use_config_args']:
                config_args = {'args': self.config_args}
                kwargs.update(config_args)
            _res = _attr(*args, **kwargs)
            return _res
        except:
            raise Exception(f'ConfigParser create {name} failed')

    def _create_attr(self, _dict, name):
        try:
            module_name = _dict['module']
            attr_name = _dict['attr']
            _module = importlib.import_module(module_name)
            _attr = getattr(_module, attr_name)
            return _attr
        except:
            raise Exception(f'ConfigParser create {name} failed')

class JsonConfigParser(ConfigParser):
    def load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)