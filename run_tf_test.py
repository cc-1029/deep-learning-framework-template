import argparse

from base import base_trainer
from utils import common_utils

parser = argparse.ArgumentParser(description="get train args")
parser.add_argument("--config", type=str, default="conf/config_tf_custom.json")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=1e-4)
args = parser.parse_args()

config_parser = common_utils.JsonConfigParser.from_cli_args(args)

trainer = config_parser.init_trainer()

trainer.train()
print('done')