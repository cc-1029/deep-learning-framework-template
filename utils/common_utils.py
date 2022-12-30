import abc
import argparse
import importlib
import json

import yaml


class ConfigParser:
    def __init__(self, config):
        self._config = config

    @classmethod
    def from_cli_args(cls, args):
        """
        Construct method from cli args

        Args:
            args: Arguments from the command line
        """
        assert isinstance(args, argparse.Namespace), 'cli args type error'
        assert 'config' in args.__dict__, 'config not in cli args'
        config = cls.load_config(cls, args.config)
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
            if 'args' in _dict:
                dict_kwargs = dict(_dict['args'])
                kwargs.update(dict_kwargs)
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


class YamlConfigParser(ConfigParser):
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.SafeLoader)
